import shap
import numpy as np
import matplotlib.pyplot as plt


def shap_analysis(model,X):

    explainer = shap.Explainer(model)

    shap_values = explainer(X)

    shap.plots.bar(shap_values)

    shap.plots.beeswarm(shap_values)