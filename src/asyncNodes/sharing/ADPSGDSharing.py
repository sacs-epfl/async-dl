import logging
import torch

from decentralizepy.sharing.Sharing import Sharing

class ADPSGDSharing(Sharing):
    """
    This class implements the averaging function for the ADPSGD algorithm.
    
    """

    def _averaging_ADPSGD(self, msg):
        """
        Averages the local model and the model received, with weight of 0.5 each

        """

        with torch.no_grad():
            total = dict()

            iteration, sender_uid = msg["iteration"], msg["sender_uid"]
            del msg["iteration"]
            del msg["sender_uid"]
            del msg["CHANNEL"]

            logging.debug("averaging model from neighbor of iteration {}".format(sender_uid, iteration))

            data = self.deserialized_model(msg)
           
            weight = 0.5

            for key, value in data.items():
                total[key] = weight * value

            for key, value in self.model.state_dict().items():
                if key in total:
                    total[key] += (1 - weight) * value
                else:
                    total[key] = (1 - weight) * value

        self.model.load_state_dict(total)
        self._post_step()


    def get_data_to_send(self, training_iteration, uid):
        self._pre_step()
        data = self.serialized_model()
        data["iteration"] = training_iteration
        data["sender_uid"] = uid
        return data
    