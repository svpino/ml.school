# Invoking The Model

After the model server is running, we can invoke it by sending a request with a sample input. The following command will send a request and output the prediction returned by the model:

```shell
just sample
```

You can see the actual command behind the `sample` recipe by opening the [`justfile`](/justfile) file. Notice we are using a simple `curl` command to send a request to the model server:

```bash
uv run -- curl -X POST http://0.0.0.0:8080/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [{"island": "Biscoe", "culmen_length_mm": 48.6, "culmen_depth_mm": 16.0, "flipper_length_mm": 230.0, "body_mass_g": 5800.0, "sex": "MALE" }]}'
```

If you are running the model using the default `backend.Local` backend, you can query the local SQLite database to ensure the new sample request was stored:

```shell
just sqlite
```

You can also run arbitrary SQL queries against the SQLite database using the following format:

```shell
uv run -- sqlite3 data/penguins.db "SELECT COUNT(*) FROM data;"
```