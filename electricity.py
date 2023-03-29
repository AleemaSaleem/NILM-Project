import torch
torch.set_default_tensor_type(torch.DoubleTensor)

# import argparse
from   config            import *
from   UKDALE_Parser     import *
from   REDD_Parser       import *
from   Refit_Parser      import *
from   Electricity_model import *
from   NILM_Dataloader   import *
from   Trainer           import *
from   time              import time
import pickle            as pkl



if __name__ == "__main__":

    args = get_args()
    setup_seed(args.seed)

    if args.dataset_code == 'redd_lf':
        args.house_indicies = [2, 3, 4, 5, 6]
        ds_parser = Redd_Parser(args)
    elif args.dataset_code == 'uk_dale':
        args.house_indicies = [1, 3, 4, 5]
        ds_parser = UK_Dale_Parser(args)
    elif args.dataset_code == 'refit':
        args.house_indicies = [2,3,16]
        args.sampling = '7s'
        ds_parser = Refit_Parser(args)

    model = ELECTRICITY(args)
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)

model.to(device)
    
    trainer = Trainer(args,ds_parser,model)

    #Training Loop
    start_time = time()
    if args.num_epochs > 0:
        try:
            model.load_state_dict(torch.load(os.path.join(trainer.export_root, 'best_acc_model.pth'), map_location='cpu'))
            print('Successfully loaded previous model, continue training...')
        except FileNotFoundError:
            print('Failed to load old model, continue training new model...')
        trainer.train()

    end_time = time()

    training_time = end_time-start_time
    print("Total Training Time: " + str(training_time/60) + "minutes")

    #Testing Loop
    args.validation_size = 1.
    x_mean = trainer.x_mean.detach().cpu().numpy()
    x_std  = trainer.x_std.detach().cpu().numpy()
    stats  = (x_mean,x_std)
    if args.dataset_code == 'redd_lf':
        args.house_indicies = [1]
        ds_parser = Redd_Parser(args, stats)
    
    elif args.dataset_code == 'refit':
        args.house_indicies = [5]
        ds_parser = Refit_Parser(args)
    elif args.dataset_code == 'uk_dale':
        args.house_indicies = [2]
        ds_parser = UK_Dale_Parser(args, stats)


    dataloader = NILMDataloader(args, ds_parser)
    _, test_loader = dataloader.get_dataloaders()
    mre, mae, acc, prec, recall, f1 = trainer.test(test_loader)
    print('Mean Accuracy:', acc)
    print('Mean F1-Score:', f1)
    print('MAE:', mae)
    print('MRE:', mre)

    results = dict()

    results['args']          = args
    results['training_time']  = training_time/60
    results['best_epoch']    = trainer.best_model_epoch 
    results['training_loss'] = trainer.training_loss
    results['val_rel_err']   = trainer.test_metrics_dict['mre']
    results['val_abs_err']   = trainer.test_metrics_dict['mae']
    results['val_acc']       = trainer.test_metrics_dict['acc']
    results['val_precision'] = trainer.test_metrics_dict['precision']
    results['val_recall']    = trainer.test_metrics_dict['recall']
    results['val_f1']        = trainer.test_metrics_dict['f1']

    results['label_curve']   = trainer.y_curve
    results['e_pred_curve']  = trainer.y_pred_curve
    results['status_curve']  = trainer.status_curve
    results['s_pred_curve']  = trainer.s_pred_curve

    fname = trainer.export_root.joinpath('results.pkl')
    pkl.dump(results,open( fname, "wb" )) 

