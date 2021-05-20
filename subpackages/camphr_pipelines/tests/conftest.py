@pytest.fixture(scope="session", params=["cuda", "cpu"])
def device(request):
    if request.param == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("cuda is required")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def trf_model_config(lang, trf_name_or_path, device) -> Dict[str, Any]:
    return yaml_to_dict(
        f"""
    lang:
        name: {lang}
        optimizer:
            class: torch.optim.SGD
            params:
                lr: 0.01
    pipeline:
        {TRANSFORMERS_MODEL}:
          trf_name_or_path: {trf_name_or_path}
    """
    )


@pytest.fixture(scope="module")
def nlp_trf_model(lang, trf_name_or_path, device):
    _nlp = create_model(trf_model_config)
    _nlp.to(device)
    return _nlp
