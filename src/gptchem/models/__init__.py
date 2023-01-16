import pystow


def get_e_pi_pistar_model_data():
    """Return a GPR model that predicts the pi-pi* transition wavelength of the E isomer of a photoswitch."""
    return pystow.module("gptchem").ensure(
        "e_pi_pistar_model",
        url="https://www.dropbox.com/s/ulj6javsa2gvnml/gpr_e_iso.joblib?dl=1",
    )


def get_z_pi_pistar_model_data():
    """Return a GPR model that predicts the pi-pi* transition wavelength of the Z isomer of a photoswitch."""
    return pystow.module("gptchem").ensure(
        "z_pi_pistar_model",
        url="https://www.dropbox.com/s/gyl4r25xmuggpx7/gpr_z_iso.joblib?dl=1",
    )
