from attack import Ens_FGSM, Ens_IFGSM, Ens_MIFGSM, Ens_DIFGSM, Ens_TIFGSM
from attack import AdaEA_FGSM, AdaEA_IFGSM, AdaEA_MIFGSM, AdaEA_DIFGSM, AdaEA_TIFGSM
from attack import SVRE_FGSM, SVRE_IFGSM, SVRE_DIFGSM, SVRE_MIFGSM, SVRE_TIFGSM
from attack import Ours_FGSM, Ours_IFGSM, Ours_MIFGSM, Ours_DIFGSM, Ours_TIFGSM

from defense.RP import *
from defense.Bit_red import *
from defense.JPEG import *
from defense.NRP import *
from defense.RS import *
from defense.FD import *
from defense.HGD import *

def get_defense():
    rp = Randomization()
    bit_red = BitDepthReduction()
    jpeg = JPEG_compression()
    nrp = NRP_defense()
    return {
        'RP': rp,
        'Bit_red': bit_red,
        'JPEG': jpeg,
        'NRP': nrp,
    }
