# Re-implement-Memorizing-Normality-to-Detect-Anomaly
Re-implement paper  Memorizing Normality to Detect Anomaly:Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection


    uscd-ped2  
        -training
            -frames
                -01
                    -000.jpg
                    -001.jpg
                      .
                      .
                      .
                -02
                    -000.jpg
                      .
                      .
                      .
        -testing
            -same as training
        -ped2.mat
***

    Avenue  same as uscd-ped2  recommend rename some img file in Avenue from xxx.jpg to xxxx.jpg  
      -training
        -frames
          -01
            -0000.jpg
            -0001.jpg
              .
              .
              .
           -02
            -0000.jpg
              .
              .
              .
      -testing
        -same as training
      -testing_label_mask
        -1_label.mat
        -2_label.mat
          .
          .
          .

  ***
 |Hyperparameter|              |
 | ---------- | :-----------:  |            
 |optimizer|sgd|
 |init lr|0.01|
 |lr schedule|step /10  after 10 epoch|
 |total epochs|40|
 
  AutoEncoder model preformence 
 |AUC|       in each video           | average in each video    |total auc|
 | ---------- | :-----------:  | :-----------:  | :-----------:  |
 |Avenue|                   |                       |     0.75   |
 |USCDped2|                         |        |              0.79   |
 
  
  AutoEncoder plus memory model + entropy loss   preformence 
 |AUC|       in each video           | average in each video    |total auc|
 | ---------- | :-----------:  | :-----------:  | :-----------:  |
 |Avenue|                   |                       |        |
 |USCDped2|                         |        |                |

