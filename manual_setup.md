# LineupIQ -- Manual Setup Guide

Complete this guide before any implementation code is written. Every section ends with
a verification step so you know it worked.

---

## Table of Contents

1. [Snowflake Account](#1-snowflake-account)
2. [Local Machine Prerequisites](#2-local-machine-prerequisites)
3. [Kaggle Account and API Token](#3-kaggle-account-and-api-token)
4. [Data Acquisition](#4-data-acquisition)
5. [Environment Variables and Secrets](#5-environment-variables-and-secrets)
6. [Snowflake Object Setup (SQL)](#6-snowflake-object-setup-sql)
7. [Cost Guardrails](#7-cost-guardrails)
8. [Cortex Access Verification](#8-cortex-access-verification)
9. [Git and Repo Hygiene](#9-git-and-repo-hygiene)
10. [Compliance and Licensing](#10-compliance-and-licensing)
11. [Full Validation Checklist](#11-full-validation-checklist)

---

## 1. Snowflake Account

### What you need

A Snowflake account with **Enterprise Edition** (required for Feature Store and
Model Registry). Cortex AI functions (`AI_COMPLETE`, Cortex Search, Cortex Analyst)
are available on Enterprise and above.

### Option A -- New trial account (recommended for starting fresh)

1. Go to <https://signup.snowflake.com/>
2. Sign up with a valid email. No credit card is required.
3. **Edition**: Select **Enterprise**.
4. **Cloud provider / region**: Select **AWS** and one of these regions (confirmed
   Cortex AI + Cortex Search support):
   - `US West (Oregon)` -- `us-west-2`
   - `US East (N. Virginia)` -- `us-east-1`

   If your preferred region doesn't support a specific model, Snowflake's cross-region
   inference can route requests to a supported region. You can enable this later with:
   ```sql
   ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';
   ```
5. Complete email verification and set your password.

The trial gives you **$400 in credits over 30 days**. Enterprise features consume
slightly more credits than Standard, but $400 is more than enough for this project.
After 30 days the account suspends (no charges) until you add a payment method.

### Option B -- Existing account

- Confirm edition: run `SELECT CURRENT_VERSION();` and check
  `SHOW ORGANIZATION ACCOUNTS;` or your Snowsight account dropdown for "Enterprise."
- Confirm region supports Cortex, or enable cross-region inference as above.

### Record these values (you'll need them in step 5)

| Value | Example | Where to find it |
|-------|---------|-------------------|
| Account identifier | `xy12345.us-west-2.aws` | Snowsight > Account menu > Copy account URL, extract identifier |
| Username | `LINEUPIQ_ADMIN` | You'll create this in step 6 (or use existing) |
| Password | *(secure)* | Set during signup |

**Verify**: Log in to Snowsight at `https://app.snowflake.com`. You should see the
Snowsight dashboard.

---

## 2. Local Machine Prerequisites

### Python

Install **Python 3.10 or 3.11** (Snowpark requires one of these; 3.12 is preview
only and may have compatibility issues with `snowflake-ml-python`).

Check your version:
```powershell
python --version
```

If you need to install/manage versions, use either:
- **conda** (recommended): <https://docs.conda.io/en/latest/miniconda.html>
- **pyenv-win**: <https://github.com/pyenv-win/pyenv-win>

### Virtual environment

Create an isolated environment for this project:

**With conda:**
```powershell
conda create -n lineupiq python=3.11 -y
conda activate lineupiq
```

**With venv:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Python packages

Install all required packages:
```powershell
pip install "snowflake-snowpark-python[pandas]" "snowflake-ml-python>=1.5.4" kaggle lightgbm scikit-learn mapie numpy pandas pyarrow matplotlib plotly
```

Optional packages (install when needed):
```powershell
pip install nba-on-court nba_api pbpstats
```

### Package purpose reference

| Package | Purpose |
|---------|---------|
| `snowflake-snowpark-python[pandas]` | Snowflake Python session, DataFrame API, pandas interop |
| `snowflake-ml-python>=1.5.4` | Feature Store, Model Registry, ML APIs in Snowflake |
| `kaggle` | CLI tool for downloading Kaggle datasets |
| `lightgbm` | Gradient boosting for P(make) shot model |
| `scikit-learn` | Preprocessing, calibration, clustering, evaluation metrics |
| `mapie` | Conformal prediction intervals for EPSA uncertainty |
| `numpy`, `pandas`, `pyarrow` | Core data manipulation |
| `matplotlib`, `plotly` | Visualization and shot charts |
| `nba-on-court` | Lineup reconstruction validation oracle |
| `nba_api` | NBA.com API client for player/team metadata |
| `pbpstats` | Enhanced play-by-play with possession context |

**Verify**: Run this in Python (with your env activated):
```python
import snowflake.snowpark
import snowflake.ml
import kaggle
import lightgbm
import sklearn
import mapie
print("All packages imported successfully")
print(f"Snowpark: {snowflake.snowpark.__version__}")
print(f"Snowflake ML: {snowflake.ml.__version__}")
```

---

## 3. Kaggle Account and API Token

1. Go to <https://www.kaggle.com> and sign in (or create a free account).
2. Navigate to **Settings** (click your profile icon > Settings), or go directly to
   <https://www.kaggle.com/settings>.
3. Scroll to the **API** section and click **Create New Token**.
4. This downloads a file called `kaggle.json` containing your credentials:
   ```json
   {"username":"your_kaggle_username","key":"your_kaggle_api_key"}
   ```
5. Place this file at:
   - **Windows**: `C:\Users\<YourUser>\.kaggle\kaggle.json`
   - Create the `.kaggle` directory if it doesn't exist:
     ```powershell
     mkdir "$env:USERPROFILE\.kaggle" -ErrorAction SilentlyContinue
     Move-Item .\kaggle.json "$env:USERPROFILE\.kaggle\kaggle.json"
     ```

**Verify**:
```powershell
kaggle datasets list --sort-by votes --max-size 1
```
This should return a list of datasets without authentication errors.

---

## 4. Data Acquisition

### 4a. Primary dataset -- NBA/WNBA play-by-play and shot details

**Source**: <https://www.kaggle.com/datasets/brains14482/nba-playbyplay-and-shotdetails-data-19962021>

This dataset contains play-by-play data from three sources (`stats.nba.com`,
`data.nba.com`, `pbpstats.com`) and shot details with court coordinates. Coverage:
- `stats.nba.com` PBP + shot details: 1996/97 onward
- `pbpstats.com` PBP (with possession timing): 2000/01 onward
- `data.nba.com` PBP (with court coordinates): 2016/17 onward

**Download**:
```powershell
# Create data directory
mkdir data\kaggle -ErrorAction SilentlyContinue

# Download and unzip (warning: this is a large dataset, several GB)
kaggle datasets download -d brains14482/nba-playbyplay-and-shotdetails-data-19962021 --unzip -p .\data\kaggle\
```

**Alternative download (GitHub / Google Drive)**:
If Kaggle CLI is slow or the dataset is very large, you can also use the
`shufinskiy/nba_data` GitHub repository which provides the same data in `.tar.xz`
archives per season:
- Repo: <https://github.com/shufinskiy/nba_data>
- Google Drive full archive: see link in that repo's README

**Expected files after download** (filenames follow this pattern):

| File pattern | Source | Content | Available from |
|-------------|--------|---------|----------------|
| `nbastats_YYYY.csv` | stats.nba.com | PBP events with player IDs, descriptions, scores | 1996/97 |
| `shotdetail_YYYY.csv` | stats.nba.com | Shot attempts with x/y coords, zone, distance, make/miss | 1996/97 |
| `pbpstats_YYYY.csv` | pbpstats.com | PBP with possession timing, start type, FG counts | 2000/01 |
| `datanba_YYYY.csv` | data.nba.com | PBP with action coords, offense team ID | 2016/17 |

Where `YYYY` is the starting year of the season (e.g., `2022` = 2022-23 season).

The schema for each file type is documented at:
<https://github.com/shufinskiy/nba_data/blob/main/description_fields.md>

Key columns per source:

**nbastats PBP**: `GAME_ID`, `EVENTNUM`, `EVENTMSGTYPE`, `EVENTMSGACTIONTYPE`,
`PERIOD`, `PCTIMESTRING`, `HOMEDESCRIPTION`, `VISITORDESCRIPTION`, `SCORE`,
`SCOREMARGIN`, `PLAYER1_ID`, `PLAYER1_NAME`, `PLAYER1_TEAM_ABBREVIATION`,
`PLAYER2_ID`, `PLAYER3_ID`

**shotdetail**: `GAME_ID`, `GAME_EVENT_ID`, `PLAYER_ID`, `PLAYER_NAME`, `TEAM_ID`,
`PERIOD`, `MINUTES_REMAINING`, `SECONDS_REMAINING`, `EVENT_TYPE`, `ACTION_TYPE`,
`SHOT_TYPE`, `SHOT_ZONE_BASIC`, `SHOT_ZONE_AREA`, `SHOT_ZONE_RANGE`,
`SHOT_DISTANCE`, `LOC_X`, `LOC_Y`, `SHOT_MADE_FLAG`, `GAME_DATE`, `HTM`, `VTM`

**pbpstats PBP**: `GAMEID`, `PERIOD`, `STARTTIME`, `ENDTIME`, `EVENTS`,
`DESCRIPTION`, `OPPONENT`, `FG2A`, `FG2M`, `FG3A`, `FG3M`, `TURNOVERS`,
`OFFENSIVEREBOUNDS`, `STARTSCOREDIFFERENTIAL`, `STARTTYPE`,
`SHOOTINGFOULSDRAWN`

**datanba PBP**: `GAME_ID`, `evt`, `cl`, `de`, `locX`, `locY`, `opt1`, `mtype`,
`etype`, `tid`, `pid`, `hs`, `vs`, `oftid`, `PERIOD`

### 4b. Synergy play-type data (2012-2025)

**Source**: <https://github.com/DomSamangy/NBA_Play_Types_12_25>

This dataset provides per-player, per-season play-type breakdowns from Synergy Sports
(via NBA.com). Play types include: Isolation, Transition, PRBallHandler, PRRollman,
Postup, Spotup, Handoff, Cut, OffScreen, OffRebound, Misc.

**Download**:
```powershell
mkdir data\synergy -ErrorAction SilentlyContinue

# Option 1: Clone the repo
git clone https://github.com/DomSamangy/NBA_Play_Types_12_25.git data\synergy

# Option 2: Download the CSV directly from Google Drive
# (see the README in that repo for the direct link)
```

**Key columns**: `PLAYER`, `TEAM`, `SEASON`, `PLAY_TYPE`, `POSS`, `FREQ`, `PPP`,
`PPP_PCTL`, `GP`, `PTS`, `FG`, `FGA`, `FG_PCT`, `EFG_PCT`, `SF_FREQ`,
`AND1_FREQ`, `TOV_FREQ`

### 4c. Verify data files exist

```powershell
# Check Kaggle data landed
Get-ChildItem .\data\kaggle\ | Select-Object Name, Length | Format-Table

# Check Synergy data landed
Get-ChildItem .\data\synergy\ -Filter "*.csv" | Select-Object Name, Length | Format-Table
```

You should see CSV files for multiple seasons in each directory.

---

## 5. Environment Variables and Secrets

### Create a `.env` file

Create a file called `.env` in the project root with your Snowflake credentials:

```env
# Snowflake connection
SNOWFLAKE_ACCOUNT=xy12345.us-west-2.aws
SNOWFLAKE_USER=LINEUPIQ_ADMIN
SNOWFLAKE_PASSWORD=your_secure_password_here
SNOWFLAKE_ROLE=LINEUPIQ_ADMIN
SNOWFLAKE_WAREHOUSE=LINEUPIQ_WH_XS
SNOWFLAKE_DATABASE=LINEUPIQ

# Kaggle (optional -- CLI uses ~/.kaggle/kaggle.json by default)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

### CRITICAL: Never commit secrets

Make sure `.env` is in `.gitignore` (we'll set this up in step 9). **Never** commit
`kaggle.json`, `.env`, or any file containing passwords or API keys.

### How the code will load these

The project code will use `python-dotenv` or `os.environ` to load these values.
A minimal connection test:

```python
import os
from dotenv import load_dotenv
from snowflake.snowpark import Session

load_dotenv()

connection_params = {
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "role": os.environ["SNOWFLAKE_ROLE"],
    "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
    "database": os.environ["SNOWFLAKE_DATABASE"],
}

session = Session.builder.configs(connection_params).create()
print(session.sql("SELECT CURRENT_ACCOUNT(), CURRENT_ROLE(), CURRENT_WAREHOUSE()").collect())
session.close()
```

Install `python-dotenv` if you haven't:
```powershell
pip install python-dotenv
```

**Verify**: Run the connection test above. You should see your account, role, and
warehouse printed.

---

## 6. Snowflake Object Setup (SQL)

Run these SQL statements in Snowsight (Worksheets) or via the Snowpark session.
Log in as `ACCOUNTADMIN` for the initial setup, then switch to the project role
for day-to-day work.

### 6a. Database and schemas

```sql
USE ROLE ACCOUNTADMIN;

-- Main project database
CREATE DATABASE IF NOT EXISTS LINEUPIQ;

-- Schemas following medallion architecture + supporting schemas
CREATE SCHEMA IF NOT EXISTS LINEUPIQ.BRONZE;    -- Raw CSV landing
CREATE SCHEMA IF NOT EXISTS LINEUPIQ.SILVER;    -- Typed, conformed, stints
CREATE SCHEMA IF NOT EXISTS LINEUPIQ.GOLD;      -- Analytics / model-ready tables
CREATE SCHEMA IF NOT EXISTS LINEUPIQ.ML;        -- Model outputs, predictions, scenarios
CREATE SCHEMA IF NOT EXISTS LINEUPIQ.RAG;       -- Cortex Search service objects
CREATE SCHEMA IF NOT EXISTS LINEUPIQ.EVAL;      -- Evaluation harness tables
CREATE SCHEMA IF NOT EXISTS LINEUPIQ.STAGING;   -- Internal stages for file uploads
```

### 6b. Warehouses

```sql
-- General-purpose warehouse (X-Small is sufficient for most work)
CREATE WAREHOUSE IF NOT EXISTS LINEUPIQ_WH_XS
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'LineupIQ general-purpose compute';

-- Larger warehouse for Cortex Search index builds and bulk ML inference
CREATE WAREHOUSE IF NOT EXISTS LINEUPIQ_WH_S
    WAREHOUSE_SIZE = 'SMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'LineupIQ Cortex Search builds and batch ML';
```

### 6c. Roles and grants

```sql
-- Project roles
CREATE ROLE IF NOT EXISTS LINEUPIQ_ADMIN;   -- Full access to all project objects
CREATE ROLE IF NOT EXISTS LINEUPIQ_ML;      -- ML pipeline execution
CREATE ROLE IF NOT EXISTS LINEUPIQ_APP;     -- Streamlit app read access

-- Role hierarchy
GRANT ROLE LINEUPIQ_ML TO ROLE LINEUPIQ_ADMIN;
GRANT ROLE LINEUPIQ_APP TO ROLE LINEUPIQ_ADMIN;
GRANT ROLE LINEUPIQ_ADMIN TO ROLE SYSADMIN;

-- Database grants
GRANT OWNERSHIP ON DATABASE LINEUPIQ TO ROLE LINEUPIQ_ADMIN;
GRANT USAGE ON DATABASE LINEUPIQ TO ROLE LINEUPIQ_ML;
GRANT USAGE ON DATABASE LINEUPIQ TO ROLE LINEUPIQ_APP;

-- Schema grants for LINEUPIQ_ADMIN (owns everything)
GRANT OWNERSHIP ON ALL SCHEMAS IN DATABASE LINEUPIQ TO ROLE LINEUPIQ_ADMIN;

-- Schema grants for LINEUPIQ_ML
GRANT USAGE ON SCHEMA LINEUPIQ.BRONZE TO ROLE LINEUPIQ_ML;
GRANT USAGE ON SCHEMA LINEUPIQ.SILVER TO ROLE LINEUPIQ_ML;
GRANT USAGE ON SCHEMA LINEUPIQ.GOLD TO ROLE LINEUPIQ_ML;
GRANT USAGE ON SCHEMA LINEUPIQ.ML TO ROLE LINEUPIQ_ML;
GRANT USAGE ON SCHEMA LINEUPIQ.RAG TO ROLE LINEUPIQ_ML;
GRANT USAGE ON SCHEMA LINEUPIQ.EVAL TO ROLE LINEUPIQ_ML;
GRANT USAGE ON SCHEMA LINEUPIQ.STAGING TO ROLE LINEUPIQ_ML;

-- Table-level grants for LINEUPIQ_ML
GRANT SELECT ON ALL TABLES IN SCHEMA LINEUPIQ.BRONZE TO ROLE LINEUPIQ_ML;
GRANT SELECT ON ALL TABLES IN SCHEMA LINEUPIQ.SILVER TO ROLE LINEUPIQ_ML;
GRANT SELECT ON ALL TABLES IN SCHEMA LINEUPIQ.GOLD TO ROLE LINEUPIQ_ML;
GRANT ALL ON SCHEMA LINEUPIQ.ML TO ROLE LINEUPIQ_ML;
GRANT CREATE TABLE ON SCHEMA LINEUPIQ.ML TO ROLE LINEUPIQ_ML;
GRANT CREATE MODEL ON SCHEMA LINEUPIQ.ML TO ROLE LINEUPIQ_ML;

-- Future grants (so new tables auto-inherit)
GRANT SELECT ON FUTURE TABLES IN SCHEMA LINEUPIQ.BRONZE TO ROLE LINEUPIQ_ML;
GRANT SELECT ON FUTURE TABLES IN SCHEMA LINEUPIQ.SILVER TO ROLE LINEUPIQ_ML;
GRANT SELECT ON FUTURE TABLES IN SCHEMA LINEUPIQ.GOLD TO ROLE LINEUPIQ_ML;
GRANT SELECT ON FUTURE TABLES IN SCHEMA LINEUPIQ.ML TO ROLE LINEUPIQ_APP;

-- Schema grants for LINEUPIQ_APP (read-only on results)
GRANT USAGE ON SCHEMA LINEUPIQ.GOLD TO ROLE LINEUPIQ_APP;
GRANT USAGE ON SCHEMA LINEUPIQ.ML TO ROLE LINEUPIQ_APP;
GRANT USAGE ON SCHEMA LINEUPIQ.RAG TO ROLE LINEUPIQ_APP;
GRANT SELECT ON ALL TABLES IN SCHEMA LINEUPIQ.GOLD TO ROLE LINEUPIQ_APP;
GRANT SELECT ON ALL TABLES IN SCHEMA LINEUPIQ.ML TO ROLE LINEUPIQ_APP;

-- Warehouse grants
GRANT USAGE ON WAREHOUSE LINEUPIQ_WH_XS TO ROLE LINEUPIQ_ADMIN;
GRANT USAGE ON WAREHOUSE LINEUPIQ_WH_XS TO ROLE LINEUPIQ_ML;
GRANT USAGE ON WAREHOUSE LINEUPIQ_WH_XS TO ROLE LINEUPIQ_APP;
GRANT USAGE ON WAREHOUSE LINEUPIQ_WH_S TO ROLE LINEUPIQ_ADMIN;
GRANT USAGE ON WAREHOUSE LINEUPIQ_WH_S TO ROLE LINEUPIQ_ML;

-- Cortex access: CORTEX_USER is granted to PUBLIC by default.
-- Verify it's not been revoked:
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE LINEUPIQ_ML;
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE LINEUPIQ_APP;

-- Assign project role to your user
GRANT ROLE LINEUPIQ_ADMIN TO USER <YOUR_USERNAME>;
```

Replace `<YOUR_USERNAME>` with your actual Snowflake username.

### 6d. Internal stage for file uploads

```sql
USE ROLE LINEUPIQ_ADMIN;
USE DATABASE LINEUPIQ;
USE SCHEMA STAGING;

CREATE STAGE IF NOT EXISTS RAW_FILES
    COMMENT = 'Landing stage for Kaggle CSVs and Synergy data';
```

### 6e. File formats

```sql
USE SCHEMA STAGING;

-- CSV format for Kaggle data (most files use comma delimiter, UTF-8)
CREATE FILE FORMAT IF NOT EXISTS CSV_KAGGLE
    TYPE = 'CSV'
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    SKIP_HEADER = 1
    NULL_IF = ('', 'NA', 'null', 'NULL')
    TRIM_SPACE = TRUE
    ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE
    COMMENT = 'Format for Kaggle NBA CSV files';

-- CSV format for Synergy play-type data
CREATE FILE FORMAT IF NOT EXISTS CSV_SYNERGY
    TYPE = 'CSV'
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    SKIP_HEADER = 1
    NULL_IF = ('', 'NA', 'null', 'NULL')
    TRIM_SPACE = TRUE
    ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE
    COMMENT = 'Format for Synergy play-type CSV';
```

**Verify**: Run:
```sql
SHOW SCHEMAS IN DATABASE LINEUPIQ;
SHOW WAREHOUSES LIKE 'LINEUPIQ%';
SHOW STAGES IN SCHEMA LINEUPIQ.STAGING;
SHOW FILE FORMATS IN SCHEMA LINEUPIQ.STAGING;
```

All objects should appear.

---

## 7. Cost Guardrails

This is critical for a personal/trial account. Snowflake charges by compute-second,
and Cortex services have their own credit consumption.

### 7a. Resource monitor

```sql
USE ROLE ACCOUNTADMIN;

CREATE RESOURCE MONITOR IF NOT EXISTS LINEUPIQ_MONITOR
    WITH
        CREDIT_QUOTA = 50          -- 50 credits per month (adjust as needed)
        FREQUENCY = MONTHLY
        START_TIMESTAMP = IMMEDIATELY
        NOTIFY_USERS = ('<YOUR_USERNAME>')
        TRIGGERS
            ON 50 PERCENT DO NOTIFY    -- Email at 50% usage
            ON 75 PERCENT DO NOTIFY    -- Email at 75% usage
            ON 90 PERCENT DO NOTIFY    -- Email at 90% usage
            ON 100 PERCENT DO SUSPEND  -- Suspend warehouses at limit
            ON 110 PERCENT DO SUSPEND_IMMEDIATE;  -- Force-kill at 110%

-- Attach monitor to both warehouses
ALTER WAREHOUSE LINEUPIQ_WH_XS SET RESOURCE_MONITOR = LINEUPIQ_MONITOR;
ALTER WAREHOUSE LINEUPIQ_WH_S SET RESOURCE_MONITOR = LINEUPIQ_MONITOR;
```

Replace `<YOUR_USERNAME>` with your Snowflake username.

### 7b. Cost awareness guide

| Activity | Approximate cost | Notes |
|----------|-----------------|-------|
| XS warehouse running | ~1 credit/hour | Auto-suspends after 60s idle |
| S warehouse running | ~2 credits/hour | Only used for Cortex Search builds |
| `AI_COMPLETE` calls | Per-token billing (see Snowflake credit table) | Smaller models like `mistral-7b` are cheapest |
| Cortex Search serving | Per GB/month of indexed data | Costs accrue while service is active, even with no queries |
| Cortex Search embedding | Per token during index build | One-time per build (only re-embeds changed docs) |
| Storage | ~$23/TB/month (compressed) | Negligible for this dataset |

### 7c. Cost-saving practices

- **Always use the XS warehouse** unless building Cortex Search indexes.
- **Suspend warehouses manually** when done for the day:
  ```sql
  ALTER WAREHOUSE LINEUPIQ_WH_XS SUSPEND;
  ALTER WAREHOUSE LINEUPIQ_WH_S SUSPEND;
  ```
- **Suspend Cortex Search service** during development when not actively testing
  retrieval (serving costs accrue even with no queries).
- **Use `mistral-7b` or `snowflake-arctic`** for prompt development/testing.
  Switch to larger models (`claude-3-5-sonnet`, `llama3.1-70b`) only for final
  scouting report generation.
- **Monitor usage** regularly:
  ```sql
  SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_DAILY_HISTORY
  WHERE USAGE_DATE >= DATEADD(DAY, -7, CURRENT_DATE())
  ORDER BY USAGE_DATE DESC;
  ```

**Verify**: Run:
```sql
SHOW RESOURCE MONITORS;
```
You should see `LINEUPIQ_MONITOR` with the credit quota and triggers listed.

---

## 8. Cortex Access Verification

Run these tests to confirm Cortex AI functions are available in your account and
region.

### 8a. AI_COMPLETE (LLM text generation)

```sql
USE ROLE LINEUPIQ_ADMIN;
USE WAREHOUSE LINEUPIQ_WH_XS;

-- Basic test with a lightweight model
SELECT AI_COMPLETE('snowflake-arctic', 'What is a pick and roll in basketball? Answer in one sentence.');
```

If this fails with a region error, enable cross-region inference:
```sql
USE ROLE ACCOUNTADMIN;
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';
```
Then retry.

### 8b. Available models for AI_COMPLETE

Key models you'll use in this project:

| Model | Use case | Context window |
|-------|----------|----------------|
| `snowflake-arctic` | Cheap testing, prompt iteration | ~4K tokens |
| `mistral-7b` | Fast, cheap testing | 4,096 tokens |
| `llama3.1-70b` | Medium quality, good for structured JSON output | 128K tokens |
| `claude-3-5-sonnet` | High quality scouting narratives | 200K tokens |
| `mistral-large2` | Strong reasoning, good cost/quality balance | 128K tokens |

### 8c. AI_COMPLETE with structured output

```sql
SELECT AI_COMPLETE(
    'llama3.1-70b',
    'List the top 3 NBA shot zones by league-average efficiency. Return JSON only.',
    {'temperature': 0.1, 'max_tokens': 500}
);
```

### 8d. Cortex Search placeholder test

We won't create the search service yet (that happens in Phase 5), but confirm the
function is available:
```sql
-- This will error because no service exists yet, but confirms the function is recognized
SELECT SEARCH_PREVIEW('nonexistent_service', 'test query');
```
Expected: an error about the service not existing (not a "function not found" error).

**Verify**: If steps 8a and 8c return LLM-generated text, Cortex AI is working.

---

## 9. Git and Repo Hygiene

### 9a. `.gitignore`

Create (or update) the `.gitignore` file to protect secrets and large data files:

```gitignore
# Secrets -- NEVER commit these
.env
*.env
kaggle.json

# Data files -- too large for Git
data/
*.csv
*.tar.xz
*.parquet

# Python
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
desktop.ini

# Snowflake connection cache
~/.snowsql/
```

### 9b. Verify nothing sensitive is tracked

```powershell
git status
```

Confirm `.env`, `data/`, and `kaggle.json` do NOT appear in untracked or staged
files. If they do, add them to `.gitignore` before committing anything.

---

## 10. Compliance and Licensing

### Data sources and their licenses

| Source | License | Attribution required | Notes |
|--------|---------|---------------------|-------|
| Kaggle brains14482 NBA PBP + Shots | Check dataset page (expected CC0 / public domain) | Yes -- cite dataset in README | Verify on download page |
| DomSamangy Synergy Play Types | Check repo LICENSE (expected MIT) | Yes -- cite repo | Data from NBA.com via Synergy Sports |
| `nba_api` | MIT | Cite library | Wraps NBA.com endpoints; personal use |
| `nba-on-court` | MIT | Cite library | We only use it as validation reference |
| `pbpstats` | MIT | Cite library | Optional enrichment |

### Rules

- **Do NOT scrape** Basketball Reference, ESPN, or any site that prohibits automated
  access in its Terms of Use.
- **Do NOT commit raw NBA.com API responses** to the public repo.
- **Do NOT redistribute** Kaggle datasets in the repo. Commit only code, schemas,
  and small sample/synthetic slices for testing.
- **Attribute all data sources** in the project README.

---

## 11. Full Validation Checklist

Run through this entire checklist. Every item must pass before implementation begins.

### Accounts and access
- [ ] Snowflake account exists (Enterprise Edition)
- [ ] Can log in to Snowsight at `https://app.snowflake.com`
- [ ] Kaggle account exists
- [ ] `kaggle.json` is at `~/.kaggle/kaggle.json`

### Local environment
- [ ] Python 3.10 or 3.11 is installed
- [ ] Virtual environment is created and activated
- [ ] All required packages install without errors
- [ ] `import snowflake.snowpark` works
- [ ] `import snowflake.ml` works

### Snowflake objects
- [ ] Database `LINEUPIQ` exists with all 7 schemas
- [ ] Warehouses `LINEUPIQ_WH_XS` and `LINEUPIQ_WH_S` exist
- [ ] Roles `LINEUPIQ_ADMIN`, `LINEUPIQ_ML`, `LINEUPIQ_APP` exist
- [ ] Your user has `LINEUPIQ_ADMIN` role
- [ ] Internal stage `LINEUPIQ.STAGING.RAW_FILES` exists
- [ ] File formats `CSV_KAGGLE` and `CSV_SYNERGY` exist

### Cost guardrails
- [ ] Resource monitor `LINEUPIQ_MONITOR` is active
- [ ] Monitor is attached to both warehouses
- [ ] Credit quota is set (suggested: 50/month for dev)

### Cortex access
- [ ] `AI_COMPLETE('snowflake-arctic', ...)` returns text (no permission or region error)
- [ ] Cross-region inference enabled if needed

### Data files
- [ ] Kaggle NBA PBP + shots downloaded to `./data/kaggle/`
- [ ] At least one season of `nbastats_*.csv` and `shotdetail_*.csv` files present
- [ ] Synergy play-type CSV downloaded to `./data/synergy/`

### Secrets and Git
- [ ] `.env` file exists with Snowflake credentials
- [ ] `.gitignore` excludes `.env`, `data/`, `kaggle.json`
- [ ] `git status` shows no sensitive files tracked

### Snowflake connectivity
- [ ] Python connection test succeeds (see step 5)
- [ ] Can run `SELECT CURRENT_ACCOUNT(), CURRENT_ROLE()` from Python

---

## What's Next

Once every checkbox above is checked, you're ready for Phase 1b: the
**design document** and **study packet**. These will be written before any pipeline
code, so you fully understand the system before building it.

After those are reviewed, Phase 2 (data foundation) begins with ingesting one
season of data as a validation slice.
