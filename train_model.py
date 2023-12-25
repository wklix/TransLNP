from train import MolTrain
clf = MolTrain(task='regression',
                data_type='molecule',
                epochs=200,
                learning_rate=0.00005,
                batch_size=4,
                early_stopping=5,
                split='random',
                save_path='./old/draw_fds',
                remove_hs=True,
              )
clf.fit('./train.csv')