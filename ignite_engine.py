#create new trainer

#torch import

import torch
from torch import nn

#ignite import
try: 
    import ignite
    from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
    from ignite.handlers import ModelCheckpoint
    from ignite.contrib.handlers import TensorboardLogger

except:
    raise Exception('Have you installed pytorch-ignite? if not, run !pip install pytorch-ignite')

def engine(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim,
        loss_fn,
        epochs: int, 
        device: str,
        lr_scheduler = None,
        log_interval: int = 1000,
        early_stop:bool = True,
):
    """
    A function to train a model. 
    Args:
    model: the model to be trained.
    train_dataloader: the train dataloader.
    validation_dataloader: the validation dataloader.
    optimizer: the optimizer to be used.
    loss_fn: the loss function.
    epochs: the number of epochs to train for.
    device: the device. 
    lr_scheduler: the learning rate scheduler. Defaults to None
    log_interval: the number of iterations before logging. Defaults to 1000. Use None to cancel
    early_stop: whether to stop the training early, save, and load best models. 
    """

    #create the engine
    
    trainer = create_supervised_trainer(model = model, optimizer=optimizer, loss_fn=loss_fn, device=device)

    if lr_scheduler != None:
        lr_scheduler.optimizer = optimizer
        trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: lr_scheduler.step()) #step every iteration, 
        #might need to change this if use stuff that;s not OnCycyelLr   

    @trainer.on(Events.ITERATION_COMPLETED(every = log_interval))
    def log_training_loss(trainer):
        print(f"Epoch: {trainer.state.epoch} | Iter: {trainer.state.iteration} | Loss: {trainer.state.output:.2f}")

    val_metrics = {
        'accuracy': ignite.metrics.Accuracy(),
        'loss': ignite.metrics.Loss(loss_fn)
    }

    train_eval = create_supervised_evaluator(model=model, metrics=val_metrics, device=device)
    validation_eval = create_supervised_evaluator(model=model, metrics=val_metrics, device=device)

    @trainer.on(Events.EPOCH_COMPLETED)

    def log_training_result(trainer):
        train_eval.run(train_dataloader)
        
        print(f"Epochs: {trainer.state.epoch} | Train Loss: {train_eval.state.metrics['loss']:.2f} | Train Accuracy: {train_eval.state.metrics['accuracy']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)

    def log_validation_result(trainer):
        validation_eval.run(validation_dataloader)

        print(f"Epoch: {trainer.state.epoch} | Validation Loss: {validation_eval.state.metrics['loss']:.2f} | Validation Accuracy: {validation_eval.state.metrics['accuracy']:.2f}")

    

    def score_fn(engine):
        return engine.state.metrics['loss']
    
    checkpoint = ModelCheckpoint(
        dirname='checkpoint',
        n_saved = 2,
        filename_prefix='best',
        score_function=score_fn,
        score_name= 'loss',
        global_step_transform= ignite.handlers.global_step_from_engine(trainer), #look this up on pytorchignite website #fetch the trainer state. 
        require_empty=True,
    )
    validation_eval.add_event_handler(Events.COMPLETED, checkpoint, {'model': model})

    #tensorboard logger
    tb_logger = TensorboardLogger(log_dir = 'tb-logger')

    #attach handelr, plot loss every log_interval iters
    tb_logger.attach_output_handler(
        trainer,
        event_name = Events.ITERATION_COMPLETED(every = log_interval),
        tag = 'training',
        output_transform = lambda loss: {'batch loss': loss}
    )

    #attach handlers for plotting evaluator metrics

    for tag, eval in (('training', train_eval), ('validation', validation_eval)):
        tb_logger.attach_output_handler(
            eval,
            event_name = Events.EPOCH_COMPLETED,
            tag = tag,
            metric_names = 'all',
            global_step_transform = ignite.handlers.global_step_from_engine(trainer),
        )

    #implement early stopping
    if early_stop == True:
        ignite.handlers.EarlyStopping(patience = 3, min_delta=0.005, score_function=score_fn, trainer=validation_eval, cumulative_delta= True)
    
    #run trainer.
    trainer.run(data = train_dataloader, max_epochs = epochs)

