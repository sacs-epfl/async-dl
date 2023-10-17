import logging
import torch

from decentralizepy.sharing.Sharing import Sharing


class GossipSharing(Sharing):
    """
    This class implements the averaging functions for the Gossip Learning and Super Gossip Learning algorithm .

    """
    
    def _averaging_gossip(self, msg, local_age):
        """
        Averages the local model and the model received w.r.t. their age

        Returns new local age.

        """
        
        with torch.no_grad():
            total = dict()

            iteration, sender_age = msg["iteration"], msg["age"]
            del msg["iteration"]
            del msg["age"]
            del msg["CHANNEL"]

            if sender_age == 0:
                return local_age
            

            logging.debug(
                "Averaging model from neighbor of iteration {} with age {}".format(
                    iteration, sender_age
                )
            )

            data = self.deserialized_model(msg)
            
            weight = sender_age / (sender_age + local_age)

            logging.debug("sender age is {} and my age is {}".format(sender_age, local_age))
            logging.debug("weight for averaging is {}".format(weight))


            for key, value in data.items():
                total[key] = weight * value

            for key, value in self.model.state_dict().items():
                if key in total:
                    total[key] += (1 - weight) * value 
                else:
                    total[key] = (1 - weight) * value


        self.model.load_state_dict(total)
        self._post_step()

        logging.debug("my new age is {}".format(max(local_age, sender_age)))

        return max(local_age, sender_age)



    def _averaging_gossip_queue(self, gossip_queue, local_age):
        """
        Averages the local model and all the received models included in the gossip_queue w.r.t. their age.

        Returns new local age and the number of models aggregated (not including the local one).
        
        """

        if gossip_queue.empty():
            return local_age, 0


        with torch.no_grad():
            total = dict()

            queue_size = gossip_queue.qsize()
            
            max_age = local_age
            ages_sum = local_age
                
        
            for i in range(queue_size):
                data = gossip_queue.get()
                iteration, sender_age = data["iteration"], data["age"]
                del data["iteration"]
                del data["age"]
                del data["CHANNEL"]
                
                logging.debug("Averaging model from neighbor of iteration {} and age {}".format(iteration, sender_age))

                max_age = sender_age if sender_age > max_age else max_age
                ages_sum += sender_age

                data = self.deserialized_model(data)
                
                for key, value in data.items():
                    if key in total:
                        total[key] += sender_age * value
                    else:
                        total[key] = sender_age * value


            for key, value in self.model.state_dict().items():
                if key in total:
                    total[key] += local_age * value  
                else:
                    total[key] = local_age * value


            for key in total: 
                total[key] /= ages_sum


        self.model.load_state_dict(total)
        self._post_step()

        logging.debug("my new age is {}".format(max_age))

        return max_age, queue_size


    def get_data_to_send(self, training_iteration, model_age):
        self._pre_step()
        data = self.serialized_model()
        data["iteration"] = training_iteration
        data["age"] = model_age
        return data
