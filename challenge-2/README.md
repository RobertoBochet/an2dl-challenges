# Challenge 2 - [1st ACRE Cascade Competition](https://competitions.codalab.org/competitions/27176)

with [Amedeo Carrioli](https://github.com/meme97) and Andrea Ceruti (**TheDeepDivers**)

The challenge consists of develop an image segmentation model, that segmented an image in *crop* and *weed*.

## The approach

At the beginning of the challenge, we started by studying the “Multiclass_Segmentation” notebook from professor Lattari, and tried to adapt it to our challenge. First by adapting the input masks of our problem, that were 3 rgb channels into a single channel, using the function given in the starting kit. There were problems using this approach because the network was learning only the background class, the basic resize function that we used was losing particularities of the input image, even if we were rescaling the mask at the end. Also, we noted that VOC dataset masks were made of a single channel if a numpy array is used to represent them, but they were made of different colors when we opened them up with a visualization program. The results were poor. Amedeo spent some days trying to improve it, while Roberto, who was not convinced on this approach, started a new notebook from scratch. After some days we all started working on Roberto’s notebook because it seemed more promising.

At that point we only worked on one crop of one team: Bipbip Haricot. This was because at first we wanted to see if our model learned something, and it was better to give in input images as similar as possible to each other.

Also, at the beginning we worked on Colab, but then we all moved to Kaggle because it guaranteed more available GPU hours than Colab.

We uploaded the dataset on Kaggle, so that it was easier to work with, this is the link to the dataset: [rose eval 2019 | Kaggle](https://www.kaggle.com/robertobochet/rose-eval-2019)

So, in the notebook started from scratch we made a new generatorDataset, a new customDataset class and then we started focusing on the preprocessing.

We started by applying some filters to the input image to augment the separation between weed and the background and feeding the augmented image to our network to see what types of filters were obtaining good results.

Then we used different models for the network. At first, we did some experiments with the classic Vgg encoder and with a handcrafted decoder following the example of “multiclass segmentation” getting poor results in prediction. We then exploited a custom Unet to introduce the advantages of the skip connections from [*keras_unet*](https://pypi.org/project/keras-unet/) models and we tried it by tuning its parameters, but again the results were poor.

At this point we tried the Unet from [*segmentation-models*](https://github.com/qubvel/segmentation_models) library and with some backbones as *Vgg* and *Resnet* and we fitted a single class to each instance of the notebook.

So, at this point, the results in the leaderboard were good, but only for *Bipbip*. All the others were poor (of course this, because we didn’t train the network on other team’s images). So we trained our network with images from more teams and crops, but the results were very low.

At this point we decided to leave the idea of one network able to predict on every crop and every team, so we made multiple networks, but trying to make as few as possible. We tried many combinations but the division that worked better was: two notebooks with images from both *Bipbip* and *Weedelec*, one notebook for *Mais* and one for *Haricot*.
Also one notebook for Roseau and one for *Pead*. The results were good.

In all notebooks the model was exactly the same, but of course each one was trained on different images to learn the meaningful features of each, in order to make good predictions.

At the end we merged the json files before the submission.

We did not like this approach (merge by hand the json files), but for now it was the only way to make good results.

### Image preprocessing

The idea: The assigned task required to identify plants and classify them as crop or weed, so the information contained in the image background is not relevant for the aim of the task, indeed, it may reduce the performance of the model.

So we tried to design a preprocessing function to reduce the complexity of the images before using it to train the model.

We approached the problem in the easiest way: we exploited the color space to find only the section of the image inside a specific color range. To do this we change the color space from RGB to HSV and exploiting some histogram we design the desidered function (as you can see in the file [*preprocessing.ipynb*](./preprocessing.ipynb))

### Image tiling

Trying to overcome the latest plateau we tried to implement a form of image tiling as suggested by Professor *Mattuecci* and several opinions found on the web. The idea behind that is to reduce the feature loss given by the rescaling of a big image (each image in *Weedelec* has a resolution of 17MP, without tilling we compress it in some thousand of pixels).

Due to the HW limitations we needed to implement tiling with some limitations, for example the amount tiles for an image is fixed.

So the network improved a lot. We could now use a single network to train on all the crops and images of all the teams. And we did it.

### The general purpose model

After several models of several types and different configurations we reach a good result in terms of general purpose model.

The best overall score that we reached is with a model based on **Unet** with **DenseNet121** with a tilling of **3 rows** and **4 columns** with patch size of **256x256** and **preprocessing function** seen above. You can see this model in the file [*general_purpose_model.ipynb*](./general_purpose_model.ipynb).

As said before, this is from a single network trained on all images of all teams together. This was our final result.

### Possible improvements

The preprocessing function has poor performance with some types of images (some kind of crop or weed are excluded by the image, or most of the ground are not removed), this also happens with some images in *Rouseau* dataset, where there are several exposition changes. It would for sure be possible to improve the overall performance by working further on the function implementing some kind of adaptation of it.

Due to the preprocessing function several images tiles are poor in information (most of them are totally black), so they can be excluded from the training to speed up the training.

Limited by the available resources we chose to set a fixed amount tiles for each image, this is certainly a suboptimal approach; the dimension and the distance to the crop in the pictures may change a lot between different dataset, so the amount tiles should variating in function of the single image.

About the load bottleneck: The use of Python in the loading and preprocessing tasks slow down the entire training process, so you might think to implement some preprocessing tasks (i.e. preprocessing function, resizing and tiling) in a most performance language like C++.

Also, it was possible to improve the score by using a notebook for each team or crop, and then merge the json file.
But we did not do it because we think it is better to have a single generic network able to make good predictions on all kinds of images.
