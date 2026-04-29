# Secrets Setup — Enabling History Sync and Expert-Review Submissions

The classifier ships with two GitHub-backed features that require a personal
access token (PAT):

- **Sync to repository** (History tab) — commits the current session's
  predictions log to `submissions/predictions_log.csv` in the repo.
- **Submit for review** (Classifier tab, after a result) — opens a GitHub
  Issue with the specimen's measurements + the model's prediction + the
  submitter's proposed clade label, tagged `submission` and `needs-review`.

Without a token configured, both features are visible but disabled, and the
app shows an explanatory note. The classifier itself works without a token.

---

## Step 1 — Create a fine-grained personal access token

1. Sign in to GitHub as the repository owner (the account that owns
   `Jbong17/HOYA-FLWR-AI`).
2. Go to **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens** ([direct link](https://github.com/settings/tokens?type=beta)).
3. Click **Generate new token**.
4. Fill in the form:
   - **Token name:** `hoya-classifier-streamlit`
   - **Expiration:** 1 year (set a calendar reminder to rotate)
   - **Resource owner:** your account (`Jbong17`)
   - **Repository access:** *Only select repositories* → choose
     `Jbong17/HOYA-FLWR-AI`
   - **Repository permissions:**
     - **Contents:** *Read and write* (needed to commit the log file)
     - **Issues:** *Read and write* (needed to open review submissions)
     - Leave everything else as *No access*.
5. Click **Generate token** and copy the value (it starts with
   `github_pat_…`). You will not see it again after closing the page.

> :lock: This token has write access to the repository — never paste it
> into chat, commits, or screenshots. The only place it should live is
> Streamlit Cloud's encrypted secrets store and your password manager.

---

## Step 2 — Add the token to Streamlit Cloud

1. Go to <https://share.streamlit.io/> and open the deployed app.
2. Click **Manage app** (lower right).
3. Click **Settings** → **Secrets**.
4. Paste the following, replacing the placeholder with your token:

   ```toml
   github_token = "github_pat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

5. Click **Save**. Streamlit Cloud restarts the app automatically; the
   restart takes about 30 seconds.

---

## Step 3 — Verify

1. Open the app, run any classification.
2. Open the **History** tab. Click **Sync to repository**. You should see
   "Synced. View predictions log on GitHub." — the link goes to
   `submissions/predictions_log.csv` in the repo.
3. Back in the **Classifier** tab, expand **Submit this specimen for expert
   review**, leave the proposed clade as the model's prediction, click
   **Submit for review**. You should see "Submission opened for review." —
   the link goes to a new GitHub Issue tagged `submission` `needs-review`.

If either action returns an error mentioning HTTP 403 or 404, the token
likely lacks the right repository scope or the right permissions. Regenerate
with the steps above and re-paste.

---

## Reviewing submissions

Submissions land as GitHub Issues at
<https://github.com/Jbong17/HOYA-FLWR-AI/issues?q=is%3Aissue+label%3Asubmission>.

For each one, the dataset owner (Mr. Aurigue, or his designate) should:

1. Inspect the measurements and the submitter's proposed clade.
2. If the proposed identification is sound and the measurements are
   plausible, comment **`verified`** on the issue and assign the
   `approved-for-retrain` label.
3. If the submission is unreliable (suspect measurements, mislabel, no
   voucher info), comment **`rejected`** with reasoning and close.

Approved submissions accumulate until you decide to retrain. The
retraining workflow is offline:

```
1. Pull the repo locally
2. Open Hoya_Clade_Classifier_Enhanced.ipynb
3. Append the verified submissions' measurements + labels to the training
   data
4. Re-run the notebook (LOOCV, ensemble fit, export pkl)
5. git commit hoya_clade_classifier_production.pkl + the updated dataset
6. Push to main — Streamlit Cloud auto-redeploys with the new model
```

---

## Rotating or revoking the token

- **Rotate:** generate a new token via Step 1, paste it into Streamlit
  secrets (Step 2). Then go back to GitHub PAT settings and revoke the
  old token.
- **Revoke immediately** (e.g. if the token leaked): go to GitHub PAT
  settings, click the token name, click **Revoke**. The next sync/submit
  attempt will fail until a fresh token is added.
