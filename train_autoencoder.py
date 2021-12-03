from data_loader import Dataset
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch
from models import AutoEncoder
from loss import MaskedMSELoss
from utils import train
import joblib
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    cwd = os.getcwd()

    print("Loading User and Movie Mappings")

    user_ids = pd.read_csv(os.path.join(cwd, 'data', 'processed', 'users.csv'))['id'].tolist()

    movie_mappings_df = pd.read_csv(os.path.join(cwd, 'data', 'processed', 'movie_mappings.csv'))

    movie_mappings_dict = {}
    for (_, new_id, original_id) in movie_mappings_df.itertuples():
        movie_mappings_dict[original_id] = new_id

    print("Loading Train and Evaluation Datasets")

    # Having a fixed random state helps us get the same results (reproducible outputs)
    # 70 percent is used for training and 30 percent is used for testing
    train_users, test_users = train_test_split(user_ids, test_size=0.30, random_state=42)

    print(f'{len(train_users)} Users for Training')
    print(f'{len(test_users)} Users for Testing')

    train_dat = Dataset(os.path.join(cwd, 'data', 'processed'), 'train_user_ratings', train_users, movie_mappings_dict)

    evaluation_dat = Dataset(os.path.join(cwd, 'data', 'processed'), 'train_user_ratings', test_users,
                             movie_mappings_dict)
    # evaluation_dat = Dataset(os.path.join(cwd, 'data', 'processed'), 'evaluation_user_ratings', user_ids,
    #                          movie_mappings_dict)

    batch_size = 128
    num_epochs = 60
    # dimensions = [400]
    dimensions = [400, 200, 400]

    print("Turning the Datasets to PyTorch DataLoader")
    train_dl = DataLoader(dataset=train_dat, batch_size=batch_size, shuffle=True, num_workers=1)
    evaluation_dl = DataLoader(dataset=evaluation_dat, batch_size=batch_size, shuffle=True, num_workers=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f'Device is {device}')

    print("Creating Model and Parameters")
    model = AutoEncoder(num_features=len(movie_mappings_df), dimensions=dimensions)

    model = model.to(device)

    criterion = MaskedMSELoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training the Model")

    dimension_names = [str(dimension) for dimension in dimensions]

    checkpoint_dir = os.path.join(cwd, 'saved_models')
    model, losses, best_model_states, best_model_epoch = train(model, criterion, optimizer, train_dl, evaluation_dl,
                                                               dimension_names, checkpoint_dir, device,
                                                               num_epochs=num_epochs)

    best_model_path = os.path.join(cwd, 'saved_models', f'best_model_{"_".join(dimension_names)}.pth')
    final_model_path = os.path.join(cwd, 'saved_models', f'final_model_{"_".join(dimension_names)}.pth')
    with open(os.path.join(cwd, 'saved_models', f'model_info_{"_".join(dimension_names)}.txt'), 'w+') as f:
        info = [
            f'Number of Users for Training: {len(train_users)}',
            f'Number of Users for Testing: {len(test_users)}',
            f'Best Model Epoch: {best_model_epoch}',
            f'Final Model Epoch: {num_epochs}',
            f'Number of Epochs: {num_epochs}',
            f'Batch Size: {batch_size}'
        ]
        f.write('\n'.join(info))

    torch.save(best_model_states, best_model_path)
    torch.save(model.state_dict(), final_model_path)
    joblib.dump(losses, os.path.join(cwd, 'saved_models', 'losses.pkl'))

    print('Model Saved')
