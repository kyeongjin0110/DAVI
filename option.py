def get_loss_opt(
    opt,
):
    if opt.loss_opt == 0:
        loss_list = ['cd_loss']     
    elif opt.loss_opt == 1:
        loss_list = ['cd_loss', 'entropy_loss']
    else:
        assert(0)
        
    return loss_list