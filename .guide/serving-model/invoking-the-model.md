# Invoking The Model

After the model server is running, we can invoke it by sending a request with a sample input. The following command will send a request and output the prediction returned by the model:

```shell
just invoke
```

You can see the actual command behind the `invoke` recipe by opening the [`justfile`](/justfile) file. Notice we are using a simple `curl` command to send a request to the model server.

If the model is capturing data, we can check whether the data was stored correctly. For example, if we are using the `backend.Local` backend, we can query the SQLite database to make sure every new request and prediction is being stored. 

By default, `backend.Local` stores the data in a file named `penguins.db` located in the repository's root directory. We can display the number of samples in the SQLite database by running the following command:

```shell
uv run -- sqlite3 penguins.db "SELECT COUNT(*) FROM data;"
```

Run the above command after invoking the model a few times to see the number of samples in the database increase. You can also use the `sqlite` recipe to accomplish the same:

```shell
just sqlite
```