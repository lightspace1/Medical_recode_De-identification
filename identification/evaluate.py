
import pickle
import de_id
model_file_path = "./DATE"
test_fold_path = "TEST_SET"
model_type = "DATE"


if __name__ == "__main__":
    file = open(model_file_path, 'rb')
    model = pickle.load(file)
    iterator = de_id.read_files(test_fold_path, model_type, 200)

    for x, y in iterator:
        de_id.evalutation(model, x, y)