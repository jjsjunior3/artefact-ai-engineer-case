# Empório da Música

A Python CLI for a fictional music store. The assistant searches the in-stock catalog, explains store policies with page references, and redirects unrelated questions.

[Live CLI examples in English](examples-en.md) · [Original Portuguese excerpts](examples.md) · [Catalog selection rules](catalog-design.md)

## Run locally

Use Python 3.12 and the original challenge data directory. Keep the supplied CSV filenames and the store policy PDF. The dataset and manual are supplied separately from the public repository.

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS or Linux, or `.\.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -r requirements.txt
python cli.py "/path/to/data" "/path/to/store-policy.pdf"
```

Before launching the CLI, copy `.env.example` to `.env` in the repository root and set `OPENROUTER_API_KEY`. The file is ignored by Git. The default model is `openai/gpt-4.1-mini`. An existing environment variable takes precedence over the local file. Type `sair` to exit.

## Architecture and decisions

- **CLI and LangGraph.** The terminal keeps setup small. LangGraph makes the model-to-tool loop explicit. The graph executes catalog or policy calls, feeds their results back to the model, and stops when the model returns a final answer.
- **GPT-4.1 mini via OpenRouter.** The model supports function calling, and OpenRouter uses the same API style used in my previous policy assistant project. The model interprets the question and writes the reply. Python handles the catalog filters, counts, and price selection.
- **Catalog.** Look up named models for price and availability. For budget searches, join CSV rows by category ID and filter active products with stock. Use decimal arithmetic for prices and apply requested string type and amplification filters before counting. Show two low-price options, a median-price option, and one near the ceiling when enough matches exist. When there are zero matches within budget, show the lowest price for an available matching model, if one exists.
- **Policy retrieval.** A short section index maps topics to pages in the eight-page manual. The model selects a page, and the tool returns its text and page number. The prompt asks for the source and for missing facts before applying conditional rules.
- **Conversation and persona.** Léo introduces himself as the store's AI assistant. His tone is cheerful, informal, and professional. He explains instrument features in terms of how the customer will use them and grounds product details in the catalog. Accessory requests are politely redirected because the store specializes in instruments. He replies in the customer's language. History stays in memory for the current CLI session so follow-up questions can refer to previous turns.

## Tests

```bash
python -m unittest discover -v
```

The suite uses synthetic CSVs, a generated PDF, and a simulated model. It exercises catalog selection, budget boundaries, policy page numbering, and the graph's tool execution without API credentials. The conversation excerpts were captured separately using the live OpenRouter API.

## Limitations and next steps

The catalog is a local snapshot. The policy index is tied to the supplied PDF layout. Product-specific filters cover string type and amplification. The model can omit details or select an incorrect page, so the automated tests are complemented by live conversation checks. Individual order status is referred to store support.

### What I would build with one more week

I would extend Léo from product advice to a guided shopping flow, using the store's existing commerce APIs where available.

1. **Live inventory and product gallery.** Read current price and stock by SKU and show official product photos in the chat. Keep the CSV catalog available for local development and tests.
2. **Session cart.** Let the customer add, remove, and review instruments as they compare options. Show quantities and the running total, and recheck stock and prices before the next step.
3. **Reservation and checkout handoff.** Show the cart and ask the customer to confirm before calling the reservation API. Report a reservation only after the store confirms it, show its expiry when available, and direct the customer to the store's checkout to pay. Handle stock changes and retries without creating duplicate reservations.
4. **End-to-end evaluation.** Test the journey from recommendation to checkout, including a product selling out, a price changing, and a customer retrying a reservation. Retain the existing budget, policy, and scope checks.

If the store APIs were unavailable during that week, I would demonstrate the flow against a sandbox inventory and reservation service and document which steps require a production integration.

## Use of coding assistants

I used Codex to inspect the challenge files, implement and test the CLI, catalog and policy tools, integrate LangGraph, and document the project. I chose the CLI and shaped the recommendation rules, including the selection across price points and the starting-price fallback when nothing matches the budget. Live tests informed changes to catalog filtering and policy citation instructions.

## Optional browser interface

Install the optional dependency and run the Streamlit chat with the same paths and configuration as the CLI.

```bash
python -m pip install -r requirements-web.txt
python -m streamlit run web.py -- "/path/to/data" "/path/to/store-policy.pdf"
```

The browser chat uses the same LangGraph agent and keeps each conversation in its own Streamlit session.
