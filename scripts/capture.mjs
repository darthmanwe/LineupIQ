/**
 * Screenshot capture, scripted so the images cannot rot silently.
 *
 * Every asset in `docs/media/` is produced by this file against a running
 * stack. A screenshot pasted in by hand is a claim about the product frozen at
 * whatever moment someone happened to take it; one produced by a script is a
 * claim that can be re-checked with a single command, and that fails loudly
 * when the thing it depicts stops existing.
 *
 * The captures are chosen to show the parts a static page cannot argue for: a
 * real lineup scored with real magnitudes, the same tool refusing to size a
 * combination that has never played, a ranking with its ties left tied, a swap
 * moving a coefficient against its own pre-registered sign, and the split that
 * says most of an interval is the players rather than the model.
 *
 *   npm run capture            # against http://127.0.0.1:8788
 *   BASE=https://… npm run capture
 */

import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";

import { chromium } from "playwright";

/**
 * Player names, so the pickers can be driven the way a person drives them.
 *
 * The five slots were native `<select>`s until `/trade` needed two lineups on
 * one screen; they are comboboxes now, and `selectOption` by index no longer
 * exists. Typing a name and clicking the matching option is both what a user
 * does and a stronger check -- it exercises the search endpoint the picker is
 * built on, so a capture run fails if that endpoint stops answering.
 */
const NAMES = JSON.parse(
  await readFile(path.resolve("apps/web/public/data/players.json"), "utf8")
).players;

function nameOf(id) {
  const row = NAMES[String(id)];
  if (row === undefined) throw new Error(`no such player in the export: ${id}`);
  return row.name;
}

const BASE = process.env.BASE ?? "http://127.0.0.1:8788";
const OUT = path.resolve("docs/media");

const VIEWPORT = { width: 1400, height: 1000 };
/** 2×, so the text survives LinkedIn's re-encoding. */
const SCALE = 2;

/**
 * Denver's championship starting five: 5,775 possessions together, which
 * clears the 200-possession reportable floor by a factor of twenty-eight.
 *
 * Chosen because it is one of the ~487 groups out of 49,827 that clear that
 * floor at all — the estimability finding this whole product is organised
 * around. A hero image showing withheld magnitudes would be honest and
 * unreadable; this one shows the tool working, and the capture below shows it
 * refusing.
 */
const DENVER = {
  offense: [203484, 203932, 203999, 1627750, 1629008],
  shooter: 203999, // Jokic
  // Boston's 2023-24 starters, so the defence is a real five rather than
  // whoever the page happened to default to. The first capture had Jokic listed
  // among his own defenders -- legal input, and it looks like a bug.
  defense: [1628369, 1627759, 201950, 1628401, 204001],
};

/**
 * The five highest-volume shooters in the corpus, who have never shared a
 * floor. A real counterfactual, and a `directional` tier.
 */
const COUNTERFACTUAL = {
  offense: [201939, 2544, 203507, 203954, 1629029],
  shooter: 1629029,
  defense: [1628369, 1627759, 201950, 1628401, 204001],
};

/**
 * Choose one player in the nth picker of a panel.
 *
 * The option is clicked by id rather than by visible text: two players can
 * share a surname, and a substring click would take whichever the search
 * happened to rank first. `#…-<playerId>` is exact.
 */
async function pickPlayer(page, panel, index, playerId) {
  const field = panel.locator(".picker input").nth(index);
  await field.click();
  await field.fill("");
  await field.type(nameOf(playerId), { delay: 12 });
  const option = page.locator(`li[id$="-${playerId}"]`).first();
  await option.waitFor({ state: "visible", timeout: 15000 });
  await option.click();
}

async function pickLineup(page, { offense, shooter, defense }) {
  const panel = page.locator(".scorer__panel").first();
  for (let i = 0; i < offense.length; i += 1) {
    await pickPlayer(page, panel, i, offense[i]);
  }
  // Index 5 is the shooter, set after the five because it is repopulated from
  // whoever is on the floor. Indices 6..10 are the defence.
  await pickPlayer(page, panel, 5, shooter);
  for (let i = 0; i < defense.length; i += 1) {
    await pickPlayer(page, panel, 6 + i, defense[i]);
  }
}

async function score(page) {
  await page.getByRole("button", { name: /score this lineup/i }).click();
  await page.waitForSelector(".ranking", { timeout: 30000 });
  await page.waitForTimeout(700);
}

/**
 * Put an element's top edge at the top of the viewport.
 *
 * `scrollIntoViewIfNeeded` centres, and `scrollTo(0, 0)` goes to the page top --
 * which on the Lineup page is the *league* court, several screens above the
 * scorer. The first version of this script captured that and it looked like the
 * tool did nothing.
 */
async function alignTop(page, selector, offset = 0) {
  await page.evaluate(
    ([sel, off]) => {
      const el = document.querySelector(sel);
      if (el) window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - off);
    },
    [selector, offset]
  );
  await page.waitForTimeout(400);
}

const SHOTS = [
  {
    name: "01-lineup-scored",
    caption: "Denver's championship five, scored — the counterfactual made clickable.",
    social: true,
    async run(page) {
      await page.goto(`${BASE}/lineup/`, { waitUntil: "networkidle", timeout: 30000 });
      await pickLineup(page, DENVER);
      await score(page);
      await alignTop(page, ".scorer", 24);
    },
  },
  {
    name: "02-play-ranking",
    caption: "Ranked zones with delta-method intervals, and the ties left tied.",
    clip: ".ranking",
    async run(page) {
      await page.goto(`${BASE}/lineup/`, { waitUntil: "networkidle", timeout: 30000 });
      await pickLineup(page, DENVER);
      await score(page);
      await alignTop(page, ".ranking", 12);
    },
  },
  {
    name: "03-refusal",
    caption: "The same tool, on five who have never played together: direction only.",
    async run(page) {
      await page.goto(`${BASE}/lineup/`, { waitUntil: "networkidle", timeout: 30000 });
      await pickLineup(page, COUNTERFACTUAL);
      await score(page);
      await alignTop(page, ".priced", 340);
    },
  },
  {
    name: "07-compare",
    caption:
      "A swap, priced in shot selection — with the contradicted pre-registration shown where it bites.",
    async run(page) {
      await page.goto(`${BASE}/trade/`, { waitUntil: "networkidle", timeout: 30000 });
      const panel = page.locator(".compare__panel").first();
      for (let i = 0; i < DENVER.offense.length; i += 1) {
        await pickPlayer(page, panel, i, DENVER.offense[i]);
      }
      await pickPlayer(page, panel, 5, DENVER.shooter);
      await page.getByRole("button", { name: /^another five$/i }).click();
      // Set all five of the comparison slots explicitly, so the two lineups
      // differ by exactly one player and the response carries a `swap`. The
      // default right-hand five already differs in one slot; changing a second
      // one without resetting it would silently make this a two-player trade,
      // which is a different and much less legible question.
      //
      // Gordon out, a high-volume perimeter shooter in. This is the capture
      // that shows `spacing_x_three` moving against its own pre-registered
      // sign, which is the finding the whole page is organised around.
      // The shooter's slot on the comparison side is deliberately locked -- he
      // has to stay on both floors or the answer is a between-player comparison
      // wearing a lineup-effect label -- so it is skipped rather than set.
      const swapped = DENVER.offense.map((id) => (id === 203932 ? 1628369 : id));
      for (let i = 0; i < swapped.length; i += 1) {
        if (DENVER.offense[i] === DENVER.shooter) continue;
        await pickPlayer(page, panel, 6 + i, swapped[i]);
      }
      await page.getByRole("button", { name: /^compare$/i }).click();
      await page.waitForSelector(".compare__verdict", { timeout: 30000 });
      await page.waitForTimeout(700);
      await alignTop(page, ".compare__verdict", 24);
    },
  },
  {
    name: "08-variance-split",
    caption: "Most of the interval is the players, not the model. Said out loud, not in a tooltip.",
    async run(page) {
      await page.goto(`${BASE}/trade/`, { waitUntil: "networkidle", timeout: 30000 });
      const panel = page.locator(".compare__panel").first();
      for (let i = 0; i < DENVER.offense.length; i += 1) {
        await pickPlayer(page, panel, i, DENVER.offense[i]);
      }
      await pickPlayer(page, panel, 5, DENVER.shooter);
      await page.getByRole("button", { name: /^compare$/i }).click();
      await page.waitForSelector(".split", { timeout: 30000 });
      await page.waitForTimeout(700);
      await alignTop(page, ".split", 120);
    },
  },
  {
    name: "04-home",
    caption: "The finding, stated up front.",
    async run(page) {
      await page.goto(`${BASE}/`, { waitUntil: "networkidle", timeout: 30000 });
      await page.waitForTimeout(800);
    },
  },
  {
    name: "05-quality",
    caption: "Reconstruction validated against box-score minutes, a physical invariant.",
    async run(page) {
      await page.goto(`${BASE}/quality/`, { waitUntil: "networkidle", timeout: 30000 });
      await page.waitForTimeout(1200);
      await alignTop(page, "table", 220);
    },
  },
  {
    name: "06-trade",
    caption: "The endpoint that refuses to exist, and publishes why.",
    async run(page) {
      await page.goto(`${BASE}/trade/`, { waitUntil: "networkidle", timeout: 30000 });
      await page.waitForTimeout(800);
      // Below the live comparison, which now sits above it: this capture is of
      // the *withheld* projection and its power analysis, not of the tool.
      await alignTop(page, ".verdict", 180);
    },
  },
];

async function main() {
  await mkdir(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: VIEWPORT,
    deviceScaleFactor: SCALE,
    // Light explicitly. The pages are theme-aware, and a capture that inherited
    // the runner's preference would change appearance between machines.
    colorScheme: "light",
  });

  const page = await context.newPage();
  const failures = [];

  for (const shot of SHOTS) {
    try {
      await shot.run(page);
      const file = path.join(OUT, `${shot.name}.png`);
      if (shot.clip) {
        await page.locator(shot.clip).screenshot({ path: file });
      } else {
        await page.screenshot({ path: file });
      }
      console.log(`  ${shot.name}.png  ${shot.caption}`);

      // GitHub's social card is 1280×640 and centre-crops anything else. One
      // capture is taken at that ratio so the link preview is composed rather
      // than cropped by chance.
      if (shot.social) {
        // Its own context at 1x. GitHub's social preview wants **exactly**
        // 1280x640; the 2x device scale the other captures use produced a
        // 2560x1280 file that GitHub accepted and then rendered blank. The
        // layout is identical either way — only the pixel density differs — so
        // there is nothing to lose by matching the spec exactly.
        // Its own context at 1x. GitHub's social preview wants **exactly**
        // 1280x640; the 2x device scale the other captures use produced a
        // 2560x1280 file that GitHub accepted and then rendered blank.
        //
        // The desktop layout is kept deliberately. Dropping to a mobile width
        // makes the court bigger and the card worse: it crops mid-court, loses
        // the headline number entirely, and the wider viewport is what puts the
        // court and "+0.06 points per 100 attempts" in the same frame. Five
        // recognisable names beside a shot chart is the thing that says what
        // this is at thumbnail size.
        const cardContext = await browser.newContext({
          viewport: { width: 1280, height: 640 },
          deviceScaleFactor: 1,
          colorScheme: "light",
        });
        const card = await cardContext.newPage();
        await shot.run(card);
        // Re-aimed for the card. At 640px the `.scorer` alignment shows five
        // dropdowns and the top of a court, which is a screenshot of a form.
        //
        // Scoped to `.scorer`, because `/lineup/` renders the *league* court
        // first and a bare `.court` selector picks that one up — the card came
        // out showing league averages, which is not what the tool does.
        // Anchored on the priced number and pushed down, so the frame holds the
        // court, its legend and the headline together.
        await alignTop(card, ".scorer .priced", 430);
        await card.screenshot({ path: path.join(OUT, "social-card.png") });
        await cardContext.close();
        console.log("  social-card.png  1280x640 at 1x, for the GitHub link preview");
      }
    } catch (error) {
      // Collected rather than thrown, so one broken capture does not hide the
      // state of the others.
      failures.push(`${shot.name}: ${error.message.split("\n")[0]}`);
    }
  }

  await browser.close();

  if (failures.length > 0) {
    console.error(`\n${failures.length} capture(s) failed:`);
    for (const failure of failures) console.error(`  - ${failure}`);
    process.exitCode = 1;
  }
}

await main();
