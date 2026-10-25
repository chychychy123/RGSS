"""Build the RGSS semantic-spatial predictor and role supervision."""


def build_model(config):
    from research.model import ResearchModel
    return ResearchModel(config)
