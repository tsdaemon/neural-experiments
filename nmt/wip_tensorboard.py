# TensorBoard info
if (step + 1) % 10 == 0:
    global_step = epoch*total_batches + step + 1
    info = { 'loss': loss.item() }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, global_step)

    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), global_step)
        logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), global_step)

    tag = 'decoder_hidden_c'
    value = model.decoder_hidden_c
    logger.histo_summary(tag, value.data.cpu().numpy(), global_step)

    tag = 'decoder_hidden_h'
    value = model.decoder_hidden_h
    logger.histo_summary(tag, value.data.cpu().numpy(), global_step)