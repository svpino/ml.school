def test_step_start_loads_data(monitoring_run_with_data):
    data = monitoring_run_with_data["start"].task.data
    print(data)
    assert data.reference_dataset.stats().row_count > 0
    assert data.current_dataset.stats().row_count > 0


def test_step_start_current_dataset_is_none_if_no_data(monitoring_run_with_no_data):
    data = monitoring_run_with_no_data["start"].task.data
    assert data.reference_dataset.stats().row_count > 0
    assert data.current_dataset is None


def test_step_data_summary_generates_html_report(monitoring_run_with_data):
    data = monitoring_run_with_data["data_summary_report"].task.data
    assert hasattr(data, "html"), "The HTML report was not generated."


def test_step_data_drift_generates_html_report(monitoring_run_with_data):
    data = monitoring_run_with_data["data_drift_report"].task.data
    assert hasattr(data, "html"), "The HTML report was not generated."


def test_step_classification_generates_html_report(monitoring_run_with_data):
    data = monitoring_run_with_data["classification_report"].task.data
    assert hasattr(data, "html"), "The HTML report was not generated."


def test_step_data_summary_no_html_if_no_data(monitoring_run_with_no_data):
    data = monitoring_run_with_no_data["data_summary_report"].task.data
    assert data.html == "No production data."


def test_step_data_drift_no_html_if_no_data(monitoring_run_with_no_data):
    data = monitoring_run_with_no_data["data_drift_report"].task.data
    assert data.html == "No production data."


def test_step_classification_no_html_if_no_data(monitoring_run_with_no_data):
    data = monitoring_run_with_no_data["classification_report"].task.data
    assert data.html == "No production data."
