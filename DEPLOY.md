# Deploying the Study Online — Step by Step

## What you need
- A GitHub account (free)
- A Streamlit Community Cloud account (free, sign in with GitHub)

---

## Step 1: Put your files on GitHub

1. Go to github.com and create a new repository
   - Name it something like `mobo-study`
   - Set it to **Private** (important — keeps your study materials private)
   - Click "Create repository"

2. Upload these files to the repository:
   - `app_online.py`  ← the main study app
   - `requirements.txt`
   - `.streamlit/config.toml`

   You can drag and drop files directly on the GitHub website.

---

## Step 2: Deploy on Streamlit Community Cloud

1. Go to share.streamlit.io
2. Sign in with your GitHub account
3. Click "New app"
4. Select:
   - Repository: your `mobo-study` repo
   - Branch: `main`
   - Main file path: `app_online.py`
5. Click "Deploy"

Streamlit will build and deploy your app. Takes about 2 minutes.
You'll get a URL like: `https://yourname-mobo-study-app-online-abc123.streamlit.app`

---

## Step 3: Share the link

Send participants this URL. They can open it on any laptop/desktop browser.

Include this message:
> "Please complete this study on a laptop or desktop using Chrome or Firefox.
> Do not use a mobile phone. The study takes about 25 minutes.
> Please complete it in one sitting without interruptions.
> [YOUR LINK HERE]"

---

## Step 4: Retrieve your data

All participant data is saved in a `study_data/` folder in your repository.

To download it:
1. Go to your GitHub repository
2. Navigate to the `study_data/` folder
3. Download `results_summary.csv` — this has one row per participant with all DVs pre-computed
4. Download individual `.json` files for full raw data

Or use the Streamlit Cloud dashboard → your app → "Manage app" → view files.

---

## Condition balancing

Conditions are assigned randomly (50/50 C vs OC) per participant.
No manual changes needed between participants.

If you need to manually assign a condition (e.g. to rebalance groups):
Add `?condition=C` or `?condition=OC` to the URL:
`https://yourapp.streamlit.app?condition=C`

---

## Monitoring progress

You can check how many participants have completed by looking at `results_summary.csv`.
Each completed session adds one row. Aim for 8-10 per condition (16-20 total).

---

## Troubleshooting

**App shows error on load:** Check requirements.txt has all packages listed.

**Data not saving:** Check the `study_data/` folder exists in your repo. 
Streamlit Cloud creates it automatically on first save.

**Participant closed browser before debrief:** Their data won't be saved — 
this is expected. Just note it as an incomplete session.
