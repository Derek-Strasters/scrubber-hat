from math import floor
from math import pi
from math import sin
from math import cos
from math import sqrt
from time import sleep
from time import monotonic

from sense_hat import SenseHat


class DeltaTimer:
    def __init__(self, **kwargs):
        self.prev_time = 0
        self.pres_time = 0
        target = kwargs.get("target_time", float(1))
        if isinstance(target, float) and target > 0:
            self.target = target
        else:
            raise ValueError("Parameter 'target_time' must be a positive float")

    def set_target_time(self, target_time):
        self.target = target_time

    def start(self):
        self.prev_time = monotonic()

    def restart(self):
        self.start()

    def delta(self):
        """Returns the time since delta was last checked or timer was restarted"""
        pres_time = monotonic()
        ret = pres_time - self.prev_time
        self.prev_time = pres_time
        return ret

    def deltarget(self, **kwargs):
        positive_only = kwargs.get("positive_only", False)
        positive_only = kwargs.get("p", positive_only)
        time_to_targ = self.target - self.delta()
        return time_to_targ if not positive_only else time_to_targ if time_to_targ > 0 else 0

    def target_sleep(self):
        sleep(self.deltarget(p=1))


class Bowl:
    def __init__(self, x_1, y_1, x_2, y_2, gravity=9.80665):
        self.x_1 = x_1
        self.y_1 = y_1
        self.x_2 = x_2
        self.y_2 = y_2
        self.gravity = gravity


def get_accelerometer_mpss(ahat):
    """Test"""
    accel_in_g = ahat.get_accelerometer_raw()
    return {'x': accel_in_g.get('x', 0) * 9.8066, 'y': accel_in_g.get('y', 0) * 9.8066,
            'z': accel_in_g.get('z', 0) * 9.8066}


def usin(partial_unit):
    return 0.5 + sin(2 * pi * partial_unit) / 2


def husin(partial_hunit: float) -> float:
    return usin(partial_hunit / 2 + 0.75)


def ucos(partial_unit):
    return 0.5 + cos(2 * pi * partial_unit) / 2


def split_br(pix):
    pix_x = int(pix[0])
    pix_y = int(pix[1])
    rem_x = pix[0] - pix_x
    rem_y = pix[1] - pix_y

    near_pix = ((pix_x, pix_y), (pix_x + 1, pix_y),
                (pix_x, pix_y + 1), (pix_x + 1, pix_y + 1))

    factors = ((1 - rem_x) * (1 - rem_y), rem_x * (1 - rem_y),
               (1 - rem_x) * rem_y, rem_x * rem_y)

    def facfun(inp):
        return sqrt(inp)

    def make_white_tup(factor: float) -> tuple:
        ret_part = 255 * facfun(factor)
        ret = (int(ret_part), int(ret_part), int(ret_part))
        return ret

    return_tup = (
        (near_pix[0] + make_white_tup(factors[0])),
        (near_pix[1] + make_white_tup(factors[1])),
        (near_pix[2] + make_white_tup(factors[2])),
        (near_pix[3] + make_white_tup(factors[3]))
    )

    return return_tup


def aa_set_pix(ahat, pix):
    split = split_br(pix)
    for pixel in split:
        if all(7 >= x >= 0 for x in (pixel[0], pixel[1])):
            ahat.set_pixel(*pixel)


if __name__ == "__main__":
    hat = SenseHat()
    sleeper = DeltaTimer(target_time=0.001)

    while True:
        hat.clear()
        steps = 1500
        for i in range(0, steps):
            funity = i / steps
            apix = (usin(funity) * 7, ucos(funity) * 7)
            aa_set_pix(hat, apix)
            sleeper.target_sleep()
