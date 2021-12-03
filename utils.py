import numpy as np
import os
import torch
import joblib
import pandas as pd


def multi_hot_encoder(df, col, possible_values):
    new_df = df.copy()
    for value in possible_values:
        new_df.loc[new_df[col].str.contains(value), value] = 1
        new_df[value] = new_df[value].fillna(0)

    return new_df


def get_genre_ratings(encoded_movie_ratings, genres):
    user_ratings = []
    for user_id in encoded_movie_ratings['userId'].unique().tolist():
        print(f'Processing User {user_id}')
        user_rating_dict = {'userId': user_id}
        ratings = encoded_movie_ratings[encoded_movie_ratings['userId'] == user_id]
        for genre in genres:
            mean_genre_rating = round(ratings[ratings[genre] == 1.0]['rating'].mean(), 2)
            user_rating_dict['average_' + '_'.join(genre.lower().strip().split(' ')) + '_rating'] = mean_genre_rating
        user_ratings.append(user_rating_dict)

    genre_ratings = pd.DataFrame(user_ratings)
    return genre_ratings


def train(model, criterion, optimizer, train_dl, test_dl, dimension_names, checkpoint_dir, device, num_epochs=40):
    loss_values = {'evaluation_loss': [], 'train_loss': []}
    best_model = None
    best_loss = float(np.iinfo(np.int32).max)
    best_model_epoch = 0
    for epoch in range(1, (num_epochs + 1)):
        train_loss, valid_loss = [], []

        # Training the model
        model.train()
        for i, data in enumerate(train_dl, 0):
            # Because it is an Autoencoder, the input and output are the same
            inputs = labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.float()
            labels = labels.float()

            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            outputs = model(outputs.detach())
            outputs = outputs.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            current_loss_value = loss.item()

            train_loss.append(current_loss_value)
            if i % 100 == 0:
                print(f"Training - Epoch: {epoch} - Iteration {i + 1} - Loss: {current_loss_value}", flush=True)

        for i, data in enumerate(test_dl, 0):
            model.eval()
            inputs = labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.float()
            labels = labels.float()

            outputs = model(inputs)
            outputs = outputs.to(device)
            loss = criterion(outputs, labels)

            current_val_loss_value = loss.item()

            valid_loss.append(current_val_loss_value)

            if i % 100 == 0:
                print(f"Testing - Epoch: {epoch} - Iteration {i + 1} - Loss: {current_val_loss_value}", flush=True)

        mean_train_loss = np.mean(train_loss)
        mean_val_loss = np.mean(valid_loss)
        if mean_val_loss < best_loss:
            best_model_epoch = epoch
            best_loss = mean_val_loss
            best_model = model.state_dict()

        print("Epoch:", epoch, " Training Loss: ", mean_train_loss, " Valid Loss: ", mean_val_loss)

        loss_values['train_loss'].append(train_loss)
        loss_values['evaluation_loss'].append(valid_loss)

        if epoch % 10 == 0:
            best_model_path = os.path.join(checkpoint_dir, f'best_model_check_point_{"_".join(dimension_names)}.pth')
            final_model_path = os.path.join(checkpoint_dir, f'final_model_check_point_{"_".join(dimension_names)}.pth')
            with open(os.path.join(checkpoint_dir, f'model_info_check_point_{"_".join(dimension_names)}.txt'),
                      'w+') as f:
                info = [
                    f'Best Model Epoch: {best_model_epoch}',
                    f'Final Model Epoch: {num_epochs}',
                    f'Number of Epochs: {num_epochs}',
                ]
                f.write('\n'.join(info))

            torch.save(best_model_path, best_model_path)
            torch.save(model.state_dict(), final_model_path)
            joblib.dump(loss_values,
                        os.path.join(checkpoint_dir, f'losses_check_point_{"_".join(dimension_names)}.pkl'))

    return model, loss_values, best_model, best_model_epoch
