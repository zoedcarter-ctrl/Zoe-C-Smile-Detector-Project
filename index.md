# Smile Detector using Machine Learning
I made a smile detector using deep learning. I applied it to accurately classifying things as "cats" and "not cats".

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Zoe C | Academic Magnet HS | Nuclear Engineering and Deep Learning | Incoming Junior

![Headstone Image](catnotcat1.png)
  
# Final Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/F7M7imOVGug" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

For your final milestone, explain the outcome of your project. Key details to include are:
- What you've accomplished since your previous milestone
- What your biggest challenges and triumphs were at BSE
- A summary of key topics you learned about
- What you hope to learn in the future after everything you've learned at BSE



# Second Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/VMCjQQUHiyg?si=wXJ44Xh4ZkwI5vAN" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

I adapted my smile detector into a cat detector, which can identify objects/faces as a "cat" or "not cat" with a 95%+ accuracy.

To accomplish this, I changed the depth parameter so that it analyzed the images in RGB instead of grayscale, and used MobileNetV2 as the base model rather than LeNet. 

Using a preexisting model to train your network is known as transfer learning, where you freeze the base of the network and cut off the "head" (the top layers it was trained on) so that you can implement your own dataset. Using MobileNetV2 not only shortened the time it took to train but also assisted in higher accuracy. My favorite part about doing this project so far was being able to get hands-on experience with machine learning rather than using YouTube videos.

Originally, I was going to train my CNN to identify cat breeds, but the large amount of breeds in my dataset along with the uneven amounts of data for each one made it extremely inaccurate. I ended up having to scale down the ambition of my project in order to get good results.

For my final milestone, I hope to be able to reimplement the cat breed classifier as well.



# First Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/iFsVW0_bvoQ?si=i9X5aGsMF0z5Ny_w" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

I trained a convolutional neural network to detect smiling and not smiling faces. It utilizes two main parts: A Haar Classifier and a CNN.

**What is a convolutional neural network (CNN)?** A nueral network allows the computer to function analogously to the brain (Dubnov and Greer, 2023). A CNN is a type of neural network for data such as images. A neural network uses weights and biases while classifying an input for the best accuracy.

**What is a Haar Classifier?** A Haar Classifier is a tool used in machine learning, specifically with visuals. It uses hundreds of positive images and negative images to classify shapes. It checks for simple shapes and patterns for classification.

It was quite surprising how much debugging I had to do. I ran into many issues such as getting the correct files and making sure the code was operating correctly. For my next milestone, I will apply what I have learned to create a detector that can identify different breeds cats.



# Code

```c++
hi
```



# Bill of Materials

| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
| Rasperry Pi Kit | Rasperry Pi | $147.69 | <a href="https://www.amazon.com/RasTech-Raspberry-Starter-Heatsink-Screwdriver/dp/B0C8LV6VNZ/ref=sr_1_4?crid=3506HY00MCGVM&dib=eyJ2IjoiMSJ9._zkM62vSQ8p7tNr88715LdMv_qHh72Je-tkF9PXEa3chDE53QT4aZu4AGAb4ihE61QY4ZD55nKF6Fp2Kfs8t7AbafM_JrlJFfHo9OB4eAVGqa0EB-7aoBQHPmhKHZ2MW8ny-Kd44bMVlVxPlTWVk5YHIN5P3uKVqrE5Dcal0rKkHny-O6Xyb5ux2AOU6OwVbkag_bqBX66RQNRrgBuz-0pS43mcx93IZTQA9R8NaJJypYU2HAycp-XicTFmyU60a01Nfm9iuyo6B9yA8ppN3OQQyJ-NQ9xyNPxfTLwkqtng.yAYpU6outhQcZmOZhN9Wb6yTw7A85CNUbXZguGInZNg&dib_tag=se&keywords=raspberry%2Bpi%2Bkit&qid=1718848547&s=electronics&sprefix=rasbperry%2Bpi%2Bkit%2Celectronics%2C83&sr=1-4&th=1"> Link </a> |
| Picam with stand | Camera for the pi | $10.99 | <a href="https://www.amazon.com/Arducam-Raspberry-Camera-Module-1080P/dp/B07RWCGX5K/ref=sr_1_10?crid=1U9IECPRDX3WW&dib=eyJ2IjoiMSJ9.EQptXsj1i39Y9oggTYxdai89FVefBqmO-xGB4sBBTHO4SEXcCZUKpLs1pTfSI2UV6zy9s3AQs7Evflr1mgvYz1YCSz3mqc1fKoWJuY2h_sLEdwqeJmnuUHIk2vmkOBLRlXijApDdRtOGjvFpd22kZibWh01QrWXaEwqpEp-2yRu8AwtKM3-xvdpkUNxIUIbjqrSK_cZ26yCkFh88Ih6aKDnMHVzWvkGv8cZGmAsc7eT7RKndhuCD03QQCco8ZhufAfPk0RJ-nafMKigKik2-9dEEZYTcX1D5vsv4x-weTH8.wWxtDi-AjBJ6-FuY_isVSX857HXALzCvS0vuocOJ6xg&dib_tag=se&keywords=arducam%2Bpicam&qid=1747573778&s=electronics&sprefix=arducam%2Bpicam%2Celectronics%2C89&sr=1-10&th=1"> Link </a> |

# Other Resources/Examples

- Dubnov, Greer. *Deep & Shallow: Machine Learning for Music & Audio*
- [Smile Detector Tutorial] https://pyimagesearch.com/2021/07/14/smile-detection-with-opencv-keras-and-tensorflow/
- [Cat Dataset] https://www.kaggle.com/datasets/crawford/cat-dataset
- [Random Images Dataset] https://www.kaggle.com/datasets/ezzzio/random-images
