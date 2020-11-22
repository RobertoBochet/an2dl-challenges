# Challenge 1

We have to classify images in one of three category:
- All people in the photo are wearing a mask
- All people in the photo are not wearing a mask
- Some people in the photo are wearing a mask, the others not

## Approach

### Managing dataset

As first approach I have tried to reach a sufficient result exploiting a home made CCN to understand the best way to manage the datasets.

I have implemented 2 class to manage the dataset

#### `class Dataset`

This class takes care of loading in memory all the data, this approach result feasible due to the limited size of the dataset. Loading all the data in memory simplify the process of working with data. The class has also the task of splitting the training data into the training set and validation set with a coefficient of `0.2` \(no cross validation with k-fold is used\) and encoding the category with one-hot convention.

#### `class DatasetGenerator`

This class provides generator for the three datasets: It takes the dataset loaded from `Dataset` and apply on it the pre-processing function and it takes care of manipulating the training dataset for data augmentation.

### `class Experiment`

I have implemented another helper class to automatize some useful task, in order to speed up the process of testing new network configurations. The class is responsible of handling the making of result file and all the other output file like checkpoints and `tensorboard` log.

### CCN

As first approach I have tried to build a classical homemade CNN to test and to improve the helping class. I have used a simple CNN, after some experiments i found a good configuration in a depth of 8 macro layers and a dense layer of classifier of `512` nodes. To achieve a good result I have needed to add to my network a dropout layer in combination with a policy of early stopping in order to avoid overfitting.

I soon abandoned this approach in favour of transfer learning approach in order to reduce the disadvantage of a small training dataset.

### Transfer learning

I tried different type of pretrained network, all of these trained on `imagenet` dataset, among these `Xception`, `VGG16`, `VGG19` and `MobileNetv2` with different parameters and classifier configuration.

#### The naive approach

With the use of generic pretrained model after several tests I found a model based on `VGG16` to reach a score of `0.88888` on test dataset, you can find the code in `tl_vgg16.ipynb` file.

#### PT model with specific dataset

After the result with a model gotten from transfer learning, exploiting a model trained with generic dataset, I have tried to use a pretrained model trained to do a specific task similar to our problem.
So, I found several ML projects with the aim to identify people wearing a mask in a photo. I chose [Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection) from github user `chandrikadeb7` and I use his pretrained model as substitution of the VGG16 used in the previous step. It used a model based on `MobileNetv2`. Unfortunately also before several tries I have not found a configuration of the parameter and classifier that reach a better score than the previous one. This can be bound to the fact that in the original project the model works only on the preprocessed image where the face to classify is already detect with another ML algorithm of face detection.

#### Explicit order for the categories

Until this point I have treated the categories as an unordered list, but we can notice that the relation on the three categories are not the same. Indeed we can assume that a photo labeled as `nobody wearing a mask` can result in a false positive with a different probability if the right category is `all wearning a mask` or `someone wearing a mask`. So I can afferent with some confidence that

$$P(\textrm{classified as 'nobody wearing a mask'} | \textrm{is 'all wearing a mask'}) < P(\textrm{classified as 'nobody wearing a mask'} | \textrm{is 'someone wearing a mask'})$$

So, I tried to use this a priori information to enhance the quality of my model. To add this information to the model I tied to tract the categories as an ordinated list: [`nobody wearing a mask`, `somebody wearning a mask`, `all wearing a mask`].

I have found few materials to create an approach of this type with keras. I based my model on [this discussion](https://stats.stackexchange.com/questions/140061/how-to-set-up-neural-network-to-output-ordinal-data).
So, I changed the categories' codification in this way:

```
'nobody wearing a mask'     -> (0,0)
'somebody wearning a mask'  -> (1,0)
'all wearing a mask'        -> (1,1)
```

and I add a bias layer to avoid the possibility of classify some data as `(0,1)` in this way I have explicated the ordered relation among categories.

With this approach I reach a best accuracy of `0.90222` on the test dataset with a model based on `VGG19` trained on `imagenet`, you can find the code in `tl_ordered_vgg19.ipynb` file. I think that a better result could be achieved further explore this approach.