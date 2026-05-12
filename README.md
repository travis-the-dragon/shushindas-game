# Shushinda's Game


<p align="center">
    <img src="./images/shushinda_at_desk_1.png" width="300">
</p>

Shushinda Hushwhisper, born into a long line of unremarkable wizards, held no extraordinary promise. In fact, her talent lay in making the ordinary a touch chaotic. A magical accident quite early in her studies left her spells prone to targeting library materials rather than their intended subjects. This penchant for rearranging spellbooks and summoning dusty tomes was a source of constant frustration for her professors, but a secret delight for Shushinda.
Under the guise of exasperated sighs and mutterings of incompetence, a mischievous grin would frequently tug at her lips. A flick of her wand could send an entire shelf of grimoires waltzing across the room, or transform a stern treatise on the 'Dangers of Spontaneous Polymorphism' into a flock of startled pigeons. The whispers of her name down the hushed corridors of Unseen University were both a warning and a promise – Shushinda Hushwhisper was in the vicinity, and misplaced manuscripts were sure to follow.

# Instrumentation
This application is instrumented with [Weave](https://weave-docs.wandb.ai/). When running, interactions with Shushinda are traced using the `@weave.op` decorator for example the `LanguageModel.predict` function:

```
    @weave.op
    def predict(self, question: str, context: str = None):
        """Predict the response based on the input question and context.

        Args:
            question (str): The input question.
            context (str, optional): Additional context for the model.

        Returns:
            dict: Response containing the model's output and call ID.
        """
```

Which is traced in the Weights & Biases UI:

<kbd> ![W and B UI](./images/predict_trace.png) </kbd>


Feedback is captured and logged to the relevant trace:

<kbd> ![feedback](./images/feedback.png) </kbd>


Different LLMs can be used by changing the dropdown and the model versions (and prompts!) are tracked in Weave:

<kbd> ![models](./images/models.png) </kbd>


Finally, evaluations can be performed to detail and track how well Shushinda's responses adhere to her personality. This is a rather complex evaluation to show the versatility of custom scorers. 

<kbd> ![comparing evaluations](./images/compare_evals.png) </kbd>


# Installation

Uses [uv](https://docs.astral.sh/uv/) for environment management. Install it once with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then from the `app/` directory:

```bash
cd app

# Create the virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt
```

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI LLM calls and embeddings |
| `WANDB_API_KEY` | Yes | Weave tracing and evaluation |
| `WANDB_ENTITY` | Yes | Your W&B username or team name |

Copy `.env.dev` to `.env` and fill in the values before running.

## Vector Database

The app uses [ChromaDB](https://www.trychroma.com/) for local vector storage — no account or API key required. The database is persisted to `app/chroma_db/` and is created automatically on first run.

Before starting the app, seed the database with the documents in `docs/`:

```bash
cd app
uv run python ingest.py --all ../docs
```

To add a single document later:

```bash
uv run python ingest.py -f ../docs/some_words_about_dragons.txt
```

## BigQuery (optional)

To use BigQuery instead of ChromaDB, set `VECTOR_DB = "bigquery"` in `app.py` and configure your GCP project:

```bash
export MY_PROJECT=`gcloud config get-value project`
export REGION="us-central1"
export BQ_REGION="US"
export EMB_MODEL="textembedding-gecko@002"
export TF_VAR_project=$MY_PROJECT
export TF_VAR_bq_region=$BQ_REGION
```

Then apply the Terraform configuration and create the remote embedding model:

```bash
cd ./terraform
terraform init && terraform plan && terraform apply
```

```bash
read -r -d '' QUERY <<-EOQ
   CREATE OR REPLACE MODEL shushindas_stuff.embedding_model
   REMOTE WITH CONNECTION \`us.vertex_ai\`
   OPTIONS (ENDPOINT = '${EMB_MODEL}')
EOQ

bq query --use_legacy_sql=false $QUERY
```


# Running the App

```bash
cd app

# Start the app (runs on http://localhost:8080)
uv run python app.py
```

To run evaluations:

```bash
cd app
uv run python shushinda_judge.py -a quick    # 3 examples, fast iteration
uv run python shushinda_judge.py -a full     # full dataset, default model (gpt-4o-mini)
uv run python shushinda_judge.py -a models   # full dataset, gpt-4o-mini / gpt-4o / gpt-4-turbo
uv run python shushinda_judge.py -a prompts  # full dataset, standard / formal / chaotic prompt variants
uv run python shushinda_judge.py -a temps    # full dataset, temperature 0.2 / 0.7 / 1.2
```

Each run publishes results to Weave under a named evaluation, making it easy to compare runs side-by-side in the W&B UI.


# Architecture



# Backstory

**Important facts about Shushinda Hushwisper**

* Shushinda Hushwhisper, despite her chaotic magic, managed to graduate from Unseen University – much to the surprise of her professors.
* She has a peculiar fondness for rearranging library stacks, especially when she's bored or startled.
* No one can make a stern treatise on the 'Perils of Thaumaturgy' dance quite like Shushinda.
* While her spells rarely hit their intended target, the resulting effects are often far more entertaining.
* Shushinda believes that 'a little chaos never hurt anyone' – a sentiment not shared by the librarians of Unseen University.
