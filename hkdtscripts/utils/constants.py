# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:18:43 2020

@author: chais
"""



TRAIN_FILES = ['../data/week1_AB001/',  # 0
               '../data/4class_AB001/',  # 1
               '../data/character/',  # 2
               '../data/Action3D/',  # 3
               '../data/Activity/',  # 4
               '../data/arabic_voice/',  # 5
               '../data/JapaneseVowels/',  # 6

               # New benchmark datasets
               '../data/AREM/',  # 7
               '../data/gesture_phase/',  # 8
               '../data/HT_Sensor/',  # 9
               '../data/MovementAAL/',  # 10
               '../data/3class_AB020_202011_Car/',  # 11
               '../data/3class_AB020_202011_Door/',  #12
               '../data/2class_AB020_202011_Door/',  #13
               '../data/2class_AB017_202012_Door/',  # 14
               '../data/2class_AB017_202012+202101_Door/',  # 15
               '../data/New2class_AB017_202012+202101_Door/',  # 16
               '../data/2class_AB016_202012_Car/',  # 17
               '../data/5input_2class_AB016_202012_Car/',  # 18
               '../data/5input_2class_AB009_202012_Car/',  # 19
               '../data/4input_2class_AB016_Car/',  # 20
               '../data/4input_2class_AB016_202102_Car/',  # 21
               '../data/4input_2class_AB009_202012_Car/',  # 22
               '../data/Change2class_AB017_202101_Door/',  # 23
               '../data/Change9class_AB017_202101_Door/',  # 24
               '../data/5class_AB017_202012_Door/',  # 25
               '../data/6class_AB017_202012_Door/',  # 26
               '../data/2class_AB017_202102_Car/',  # 27
               '../data/1input_2class_AB016_Car/',  # 28
               '../data/2input_2class_AB016_Car/',  # 29
               '../data/2input_2class_AB017_202102_Car/',  # 30
               '../data/6class_AB016_202102_Door/',  # 31
               '../data/6class_AB017_202101_Door/',  # 32
               '../data/6class_AB009_202101_Door/',  # 33
               '../data/6class_AB020_202101_Door/',  # 34
               '../data/6class_AB021_202101_Door/',  # 35
               '../data/5class_AB013_202101_Door/',  # 36
               '../data/5class_AB014_202101_Door/',  # 37
               '../data/5class_AB022_202101_Door/',  # 38
               '../data/Change5class_AB017_202101_Door/',  # 39
               '../data/1input_2class_AB017_202103_Car/',  # 40
               '../data/1input_2class_AB017_202102_Car/',  # 41
               '../data/4class_AB013_202105_Door/',  # 42
               '../data/1day_4class_AB013_202105_Door/',  # 43
               '../data/make_samples_2class_AB017_202101_Door/',  # 44
               '../data/TEST2class_AB017_202012_Door/',  # 45
               '../data/20samples_test_Door/',  # 46
               '../data/30samples_test_Door/',  # 47
               '../data/1200_make_samples_2class_AB017_202101_Door/',  # 48
               '../data/1200_2class_AB017_202102_Door/',  # 49
               '../data/rengong_1200_2class_AB017_202103_Door/',  # 50
               ]

TEST_FILES = ['../data/week1_AB001/', # 0
              '../data/4class_AB001/', # 1
              '../data/character/', # 2
              '../data/Action3D/', # 3
              '../data/Activity/', # 4
              '../data/arabic_voice/', # 5
              '../data/JapaneseVowels/', # 6

              # New benchmark datasets
              '../data/AREM/', # 7
              '../data/gesture_phase/', # 8
              '../data/HT_Sensor/',  # 9
              '../data/MovementAAL/',  # 10
              '../data/3class_AB020_202011_Car/',  # 11
              '../data/3class_AB020_202011_Door/',  #12
              '../data/2class_AB020_202011_Door/',  #13
              '../data/2class_AB017_202012_Door/',  # 14
              '../data/2class_AB017_202012+202101_Door/',  # 15
              '../data/New2class_AB017_202012+202101_Door/',  # 16
              '../data/2class_AB016_202012_Car/',  # 17
              '../data/5input_2class_AB016_202012_Car/',  # 18
              '../data/5input_2class_AB009_202012_Car/',  # 19
              '../data/4input_2class_AB016_Car/',  # 20
              '../data/4input_2class_AB016_202102_Car/',  # 21
              '../data/4input_2class_AB009_202012_Car/',  # 22
              '../data/Change2class_AB017_202101_Door/',  # 23
              '../data/Change9class_AB017_202101_Door/',  # 24
              '../data/5class_AB017_202012_Door/',  # 25
              '../data/6class_AB017_202012_Door/',  # 26
              '../data/2class_AB017_202102_Car/',  # 27
              '../data/1input_2class_AB016_Car/',  # 28
              '../data/2input_2class_AB016_Car/',  # 29
              '../data/2input_2class_AB017_202102_Car/',  # 30
              '../data/6class_AB016_202102_Door/',  # 31
              '../data/6class_AB017_202101_Door/',  # 32
              '../data/6class_AB009_202101_Door/',  # 33
              '../data/6class_AB020_202101_Door/',  # 34
              '../data/6class_AB021_202101_Door/',  # 35
              '../data/5class_AB013_202101_Door/',  # 36
              '../data/5class_AB014_202101_Door/',  # 37
              '../data/5class_AB022_202101_Door/',  # 38
              '../data/Change5class_AB017_202101_Door/',  # 39
              '../data/1input_2class_AB017_202103_Car/',  # 40
              '../data/1input_2class_AB017_202102_Car/',  # 41
              '../data/4class_AB013_202105_Door/',  # 42
              '../data/1day_4class_AB013_202105_Door/',  # 43
              '../data/make_samples_2class_AB017_202101_Door/',  # 44
              '../data/TEST2class_AB017_202012_Door/',  # 45
              '../data/20samples_test_Door/',  # 46
              '../data/30samples_test_Door/',  # 47
              '../data/1200_make_samples_2class_AB017_202101_Door/',  # 48
              '../data/1200_2class_AB017_202102_Door/',  # 49
              '../data/rengong_1200_2class_AB017_202103_Door/',  # 50
              ]

MAX_NB_VARIABLES = [5,  # 0
                    5,  # 1
                    30,  # 2
                    570,  # 3
                    570,  # 4
                    39,  # 5
                    12,  # 6

                    # New benchmark datasets
                    7,  # 7
                    18,  # 8
                    11,  # 9
                    4,  # 10
                    4,  # 11
                    5,  # 12
                    1,  # 13
                    1,  # 14
                    1,  # 15
                    1,  # 16
                    1,  # 17
                    5,  # 18
                    5,  # 19
                    4,  # 20
                    4,  # 21
                    4,  # 22
                    1,  # 23
                    1,  # 24
                    1,  # 25
                    1,  # 26
                    1,  # 27
                    1,  # 28
                    2,  # 29
                    2,  # 30
                    1,  # 31
                    1,  # 32
                    1,  # 33
                    1,  # 34
                    1,  # 35
                    1,  # 36
                    1,  # 37
                    1,  # 38
                    1,  # 39
                    1,  # 40
                    1,  # 41
                    1,  # 42
                    1,  # 43
                    1,  # 44
                    1,  # 45
                    1,  # 46
                    1,  # 47
                    1,  # 48
                    1,  # 49
                    1,  # 50
                    ]

MAX_TIMESTEPS_LIST = [402,  # 0
                      402,  # 1
                      173,  # 2
                      100, # 3
                      337, # 4
                      91, # 5
                      26, # 6

                      # New benchmark datasets
                      480, # 7
                      214, # 8
                      5396, # 9
                      119, # 10
                      653, # 11
                      2000, # 12
                      2000, # 13
                      2000,  # 14
                      2000,  # 15
                      2000,  # 16
                      607,  # 17
                      607,  # 18
                      607,  # 19
                      1000,  # 20
                      1000,  # 21
                      1000,  # 22
                      1993,  # 23
                      1993,  # 24
                      1993,  # 25
                      1993,  # 26
                      2000,  # 27
                      2000,  # 28
                      2000,  # 29
                      2000,  # 30
                      2000,  # 31
                      2000,  # 32
                      2000,  # 33
                      2000,  # 34
                      2000,  # 35
                      2000,  # 36
                      2000,  # 37
                      2000,  # 38
                      2000,  # 39
                      2000,  # 40
                      2000,  # 41
                      2000,  # 42
                      2000,  # 43
                      2000,  # 44
                      2000,  # 45
                      2000,  # 46
                      2000,  # 47
                      1200,  # 48
                      1200,  # 49
                      1200,  # 50
                      ]


NB_CLASSES_LIST = [2, # 0
                   4, # 1
                   20, # 2
                   20, # 3
                   16, # 4
                   88, # 5
                   9, # 6

                   # New benchmark datasets
                   7, # 7
                   5, # 8
                   3, # 9
                   2, # 10
                   3, # 11
                   3, # 12
                   2, # 13
                   2,  # 14
                   2,  # 15
                   2,  # 16
                   2,  # 17
                   2,  # 18
                   2,  # 19
                   2,  # 20
                   2,  # 21
                   2,  # 22
                   2,  # 23
                   5,  # 24
                   5,  # 25
                   6,  # 26
                   2,  # 27
                   2,  # 28
                   2,  # 29
                   2,  # 30
                   6,  # 31
                   6,  # 32
                   6,  # 33
                   6,  # 34
                   5,  # 35
                   5,  # 36
                   5,  # 37
                   5,  # 38
                   5,  # 39
                   2,  # 40
                   2,  # 41
                   4,  # 42
                   3,  # 43
                   2,  # 44
                   2,  # 45
                   2,  # 46
                   2,  # 47
                   2,  # 48
                   2,  # 49
                   2,  # 50
                   ]
