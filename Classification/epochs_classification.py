import numpy as np
import sys
import torch
from tqdm import tqdm as tqdm
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math
import matplotlib.pyplot as plt
import os

# from .meter import AverageValueMeter

class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        #for metric in self.metrics:
        #    metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {name: AverageValueMeter() for name, metric in self.metrics.items()}
        metrics_meters['class_0_f1_score'] = AverageValueMeter()
        metrics_meters['class_0_precision'] = AverageValueMeter()
        metrics_meters['class_0_recall'] = AverageValueMeter()
        metrics_meters['class_0_accuracy'] = AverageValueMeter()
        metrics_meters['class_1_f1_score'] = AverageValueMeter()
        metrics_meters['class_1_precision'] = AverageValueMeter()
        metrics_meters['class_1_recall'] = AverageValueMeter()
        metrics_meters['class_1_accuracy'] = AverageValueMeter()
        metrics_meters['class_2_f1_score'] = AverageValueMeter()
        metrics_meters['class_2_precision'] = AverageValueMeter()
        metrics_meters['class_2_recall'] = AverageValueMeter()
        metrics_meters['class_2_accuracy'] = AverageValueMeter()
        metrics_meters['class_3_f1_score'] = AverageValueMeter()
        metrics_meters['class_3_precision'] = AverageValueMeter()
        metrics_meters['class_3_recall'] = AverageValueMeter()
        metrics_meters['class_3_accuracy'] = AverageValueMeter()

        y_t = []
        y_predicted = []

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose),) as iterator:
            for x, y in iterator:

                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {"loss": loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                prob_mask = F.softmax(y_pred, dim=1)
                pred_mask = prob_mask.argmax(dim=1)
                y_t.append(y)
                y_predicted.append(pred_mask)
                tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, y, mode="multiclass",num_classes=3)
                for name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(tp, fp, fn, tn, reduction="macro").cpu().detach().numpy()
                    metrics_meters[name].add(metric_value)
                
                for i in range(3):
                    for name, metric_fn in self.metrics.items():
                        class_metric_value = metric_fn(tp[:, i], fp[:, i], fn[:, i], tn[:, i], reduction="micro").cpu().detach().numpy()
                        if math.isnan(class_metric_value):
                            class_metric_value = 0.0 
                        metrics_meters[f"class_{i}_{name}"].add(class_metric_value)

                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)
                print(logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        y_t_array = torch.cat(y_t).cpu().numpy()
        y_predicted_array = torch.cat(y_predicted).cpu().numpy()
        cm = confusion_matrix(y_true=y_t_array, y_pred=y_predicted_array)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        number = len(os.listdir('directory'))
        plt.savefig('path_to_save_conf_matrix', dpi=300, bbox_inches='tight')
        plt.close()
        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction