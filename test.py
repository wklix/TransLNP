from predict import MolPredict
import pandas as pd
clf = MolPredict(load_model='./random_LDS',visual=False)
test_pred = clf.predict('./dataset/Random/test.csv')
test_results = pd.DataFrame({'pred':test_pred.flatten(),
                           'smiles':clf.datahub.data['smiles']
                            })
print(test_results.head())
test_results.to_csv("./random_LDS/random_LDS.csv")