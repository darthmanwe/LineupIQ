/**
 * The order-invariant lineup hash, in TypeScript.
 *
 * Every join on lineup identity depends on this producing byte-identical
 * output to `lineupiq.hashing.lineup_hash` in Python and to the DuckDB and
 * Snowflake expressions. A mismatch does not raise anywhere -- it returns zero
 * rows, everywhere, and looks like missing data.
 *
 * **The sort is numeric, not lexicographic.** Sorting the string forms puts
 * "1630552" before "201143" because "1" < "2", and any engine that sorts
 * numerically then disagrees. That is the single most likely way this function
 * breaks, so the sort is explicit and the parity fixture covers it.
 *
 * MD5 is used because it is what the Python side uses and this is an identity
 * function, not a security primitive. WebCrypto in Workers does not offer MD5,
 * so it is implemented here -- 60 lines against a dependency in a 10 ms budget.
 */

export const LINEUP_SIZE = 5;

/** MD5 of an ASCII string, lowercase hex. */
export function md5(input: string): string {
  const bytes = new TextEncoder().encode(input);
  const bitLength = bytes.length * 8;

  // Pad to a multiple of 64 bytes: 0x80, then zeros, then the length as a
  // little-endian 64-bit integer.
  const padded = new Uint8Array((((bytes.length + 8) >> 6) + 1) << 6);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 8, bitLength >>> 0, true);
  view.setUint32(padded.length - 4, Math.floor(bitLength / 0x100000000), true);

  // `noUncheckedIndexedAccess` types every indexed read as `number | undefined`,
  // including on typed arrays. The alternative of `?? 0` at each read would
  // silently produce a wrong digest if it ever fired, so the assertion is
  // isolated in `at` below with the reason it is safe.
  const shifts = new Int32Array([
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
  ]);
  const constants = new Int32Array(64);
  for (let i = 0; i < 64; i += 1) {
    constants[i] = Math.floor(Math.abs(Math.sin(i + 1)) * 0x100000000) | 0;
  }

  let a0 = 0x67452301 | 0;
  let b0 = 0xefcdab89 | 0;
  let c0 = 0x98badcfe | 0;
  let d0 = 0x10325476 | 0;

  const rotate = (value: number, amount: number): number =>
    (value << amount) | (value >>> (32 - amount));

  // Every index below is provably in range: `i` runs 0..63 against tables of
  // length 64, and `g` is a `% 16` result against a 16-word block. The published
  // MD5 test vectors are what actually verify that -- an off-by-one here changes
  // the digest, and the first vector fails immediately.
  const at = (table: Int32Array, index: number): number => table[index] as number;

  for (let chunk = 0; chunk < padded.length; chunk += 64) {
    const words = new Int32Array(16);
    for (let i = 0; i < 16; i += 1) words[i] = view.getInt32(chunk + i * 4, true);

    let [a, b, c, d] = [a0, b0, c0, d0];
    for (let i = 0; i < 64; i += 1) {
      let f: number;
      let g: number;
      if (i < 16) {
        f = (b & c) | (~b & d);
        g = i;
      } else if (i < 32) {
        f = (d & b) | (~d & c);
        g = (5 * i + 1) % 16;
      } else if (i < 48) {
        f = b ^ c ^ d;
        g = (3 * i + 5) % 16;
      } else {
        f = c ^ (b | ~d);
        g = (7 * i) % 16;
      }
      const temp = d;
      d = c;
      c = b;
      // Every intermediate is coerced back to int32 with `| 0`. Without it the
      // additions leave the exact-integer range of a JS number and the digest
      // silently diverges from every other implementation.
      b = (b + rotate((a + f + at(constants, i) + at(words, g)) | 0, at(shifts, i))) | 0;
      a = temp;
    }
    a0 = (a0 + a) | 0;
    b0 = (b0 + b) | 0;
    c0 = (c0 + c) | 0;
    d0 = (d0 + d) | 0;
  }

  const hex = (value: number): string => {
    let out = "";
    for (let i = 0; i < 4; i += 1) {
      out += ((value >>> (i * 8)) & 0xff).toString(16).padStart(2, "0");
    }
    return out;
  };
  return hex(a0) + hex(b0) + hex(c0) + hex(d0);
}

/**
 * Canonical form of a five-man group: numerically sorted ids, comma joined.
 *
 * Exported separately from the hash so a test can assert the ordering without
 * going through MD5, which is where a sort bug would otherwise hide behind a
 * digest.
 */
export function canonicalLineup(playerIds: readonly number[]): string {
  return [...playerIds].sort((a, b) => a - b).join(",");
}

export function lineupHash(playerIds: readonly number[]): string {
  if (playerIds.length !== LINEUP_SIZE) {
    throw new Error(`a lineup is ${LINEUP_SIZE} players, got ${playerIds.length}`);
  }
  if (new Set(playerIds).size !== LINEUP_SIZE) {
    throw new Error("a lineup cannot contain the same player twice");
  }
  return md5(canonicalLineup(playerIds));
}
