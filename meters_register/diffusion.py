from Storage_and_Meter.general_metrics import AverageValueMeter
from Storage_and_Meter.metric_container import MeterInterface


def diffusion_meters():
    meters = MeterInterface()
    with meters.focus_on("tra"):
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter(
            "loss", AverageValueMeter()
        )

    return meters