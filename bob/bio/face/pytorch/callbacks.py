from pytorch_lightning.callbacks import Callback

import logging
import os
from bob.bio.base.script.vanilla_biometrics import vanilla_biometrics
import bob.bio.base
import bob.measure

logger = logging.getLogger(__name__)


class VanillaBiometricsCallback(Callback):
    def __init__(self, config, output_path, name="vanilla-biometrics", fmr=0.001):
        """
        Callback that calls `bob bio pipelines vanilla-biometrics` at every `on_epoch_end`.
        FNMR@FMR=fmr is reported at every epoch
        

        Parameters
        ----------
           
           config: str
             Path containing the `bob bio pipelines vanilla-biometrics` input script.
             Please, check :any:`bob.bio.base.vanilla_biometrics_intro` on how to setup the 

           output_path: str
             Path where the checkpoiny is being written
             
           fmr: float
              False match rate threshold that will be used to compute FNRM

        """
        self.config = config
        self.fmr = fmr
        self.output_path = output_path
        self.scores_dev = os.path.join(output_path, "scores-dev")
        super(VanillaBiometricsCallback, self).__init__()

    def on_train_epoch_end(self, epoch, logs=None):
        logger.info(f"Run vanilla biometrics {epoch}. Input script: {self.config}")

        vanilla_biometrics.main(
            [self.config],
            prog_name="bob bio pipelines vanilla-biometrics",
            standalone_mode=False,
        )

        neg, pos = bob.bio.base.score.load.split_four_column(self.scores_dev)
        far_thres = bob.measure.far_threshold(neg, pos, self.fmr)
        fmr, fnmr = bob.measure.fprfnr(neg, pos, far_thres)

        self.log(f"validation/fnmr@fmr={fmr}", fnmr)

