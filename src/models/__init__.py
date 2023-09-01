import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .ddpm_model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

def create_CD_model(opt):
    from .cd_model import CD as M
    m = M(opt)
    logger.info('Cd Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

def create_SS_model(opt):
    from .seg_model import SS as M
    m = M(opt)
    logger.info('SS Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

    
    

