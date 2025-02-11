from pykeen.pipeline import pipeline

pipeline_result = pipeline(
    training='',
    testing='',
    model="TransE",
    # Training configuration
    training_kwargs=dict(
        num_epochs=50,
        use_tqdm_batch=False,
    ),
    # Runtime configuration
    random_seed=1234,
    device="gpu",
)

pipeline_result.save_to_directory('../InstanceLLMatcher/models/inmuebles_transh.pykeenmodel')
