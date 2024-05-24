def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'i2i_mamba_many':
        from .i2i_mamba_many import I2IMamba_model
        model = I2IMamba_model()

    elif opt.model == 'i2i_mamba_one':
        from .i2i_mamba_one import I2IMamba_model
        model = I2IMamba_model()
    
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
