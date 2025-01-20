def test_step_start_loads_data(monitoring_run):
    data = monitoring_run["start"].task.data
    assert len(data.reference_data) > 0
    assert len(data.current_data) > 0


def test_step_test_suite_generates_html_report(monitoring_run):
    data = monitoring_run["test_suite"].task.data
    assert hasattr(data, "html"), "The HTML report was not generated."


def test_step_data_quality_report_generates_html_report(monitoring_run):
    data = monitoring_run["data_quality_report"].task.data
    assert hasattr(data, "html"), "The HTML report was not generated."


def test_step_test_accuracy_score_generates_html_report(monitoring_run):
    data = monitoring_run["test_accuracy_score"].task.data
    assert hasattr(data, "html"), "The HTML report was not generated."


def test_step_target_drift_report_generates_html_report(monitoring_run):
    data = monitoring_run["target_drift_report"].task.data
    assert hasattr(data, "html"), "The HTML report was not generated."


def test_step_classification_report_generates_html_report(monitoring_run):
    data = monitoring_run["classification_report"].task.data
    assert hasattr(data, "html"), "The HTML report was not generated."
