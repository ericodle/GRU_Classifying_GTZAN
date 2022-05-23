<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="https://github.com/ericodle/Genre-Classification-Using-LSTM/blob/main/GRU_CM.jpg" alt="Logo" width="400" height="350">
  </a>

<h3 align="center">Music Genre Classification Using Neural Networks</h3>

  <p align="center">
  In this project, we explore various artifical neural network (ANN) approaches to achieve near-human accuracy in a music genre classification task. By converting raw .wav audio input into an array of MFCC values, we are able to achieve our best result (90.7% accuracy) using a Gated Recurrent Unit (GRU) model written in PyTorch. This repository serves as a source of supplementary material for a music genre classification conference paper currently under review. We herein archive our Python scripts and provide a sample Google Colab notebook for the reference of anyone interested.
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Report Bug</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Request Feature</a>
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## About this Project

This was my first project with Professor Rebecca Lin (Feng Chia University, Taiwan) during the Taiwan Experience Exchange Program. Classifying music genre was my initial experience both in writing/training ANNs from scratch as well as in audio signal analysis, and I had a lot of catching up to do. Through a wealth of online resources, particularly the MFCC tutorials provided by [Valerio Velardo](https://github.com/musikalkemist), we were able to train models with decent generaliation and test classification accuracy. 

Our results are based on the [GTZAN](http://marsyas.info/index.html) music genre dataset, which provides 10 human-classified genre folders: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. Each genre folder contains 100 30-second audio clips of genre-specific songs in .wav format. Following previous work in this field, we extracted [Mel-frequency cepstral coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), or MFCCs, from each audio clip, divided the entire shuffled set into an 80:10:10 train/validation/test split, and played around with multi-layer perceptron, convolutional, and recurrent networks/hyperparamteters until we got a model that achieved at least 90% accuracy.

We hope this project inspires you to contribute to our project, incorporate our tools, and play around with ANN models yourself! 


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This project involves the training of artificial neural networks, so we recommend using either a powerful GPU or a Google Colab notebook.


<!-- USAGE EXAMPLES -->
## Walkthrough

Walkthrough video coming soon.

<p align="right">(<a href="#top">back to top</a>)</p>


## Content

- [ ] MFCC_primer.ipynb

blues.00000.wav. Music clip "One Bourbon, One Scotch, One Beer" from GTZAN.

- [ ] MFCC_primer.ipynb

coming soon!

- [ ] MFCC_extraction.py

coming soon!

- [ ] network_train_and_test.py

coming soon!

- [ ] models.py

coming soon!

- [ ] sample_run.ipynb

coming soon!

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions make the open source community great. Everyone has a unique combination of skills and experience. Your input is **highly valued**.
If you have ideas for improvement, please fork the repo and create a pull request. 
If this is your first pull request, just follow the steps below:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the GNU Lesser General Public License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
