from train import MolTrain
clf = MolTrain(task='regression',
                data_type='molecule',
                epochs=200,
                learning_rate=0.00005,
                batch_size=4,
                early_stopping=5,
                save_path='./random_LDS',
                remove_hs=True,
              )
clf.fit('./dataset/Random/train.csv')