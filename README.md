## AVL Tree Implementation for SpartaHack 9 Application Review

## Description

This project features a fully-implemented AVL tree structure, intricately integrated into a machine learning solution for automating the SpartaHack 9 application review process. The project combines this advanced data structure with the k-Nearest Neighbors algorithm, aiming to optimize the selection process for efficiency, fairness, and bias elimination.

## Project Files

- solution.py contains the full AVL tree implementation as well as the application functions.



## Background

Facing the challenge of manually reviewing a large number of applications, this project builds on the successes and lessons learned from SpartaHack 8. It introduces a data-driven strategy, enhancing the review process through machine learning and complex data structures.

AVL Tree Implementation Details

Complete AVL Tree Structure: The AVLTree class embodies a comprehensive implementation of AVL trees, facilitating efficient data management. Key methods include:
insert, remove, search: Core operations for managing tree nodes.
left_rotate, right_rotate, rebalance: Essential for maintaining tree balance.
inorder, preorder, postorder, levelorder: Traversal methods for various use cases.
visualize: Generates a visual representation of the tree structure.
Integration with Machine Learning: The NearestNeighborClassifier leverages the AVL tree for efficient data storage and retrieval, crucial for the k-Nearest Neighbors algorithm.
Machine Learning Application

NearestNeighborClassifier: A one-dimensional classifier using AVL tree lookups, designed for high-resolution data classification.
Training and Prediction: Incorporates methods like fit and predict for training the classifier and making predictions based on the most common labels in training data.

## Key Features

Efficient Data Handling: AVL tree ensures optimal data management with logarithmic operation time.
Fairness and Precision: The system leverages past data for unbiased decision-making in application reviews.
Automated and Streamlined Process: Facilitates a faster, more efficient, and bias-free review process.

## Technologies Used

Python
AVL Trees
k-Nearest Neighbors Algorithm

## Setup and Installation

Clone the repository to your local machine.
Ensure Python is installed on your system.
Navigate to the project directory.
Run solution.py to execute the application.

## Usage

The system processes application data, classifying each application based on its similarity to past applicants using the AVL tree for data management and the k-Nearest Neighbors algorithm for predictive analysis. The outcome (accept or reject) for each application is predicted using the k-Nearest Neighbors algorithm.


## Contributions

Developed by: Shams Alkhalidy
Inspired by SpartaHack 8's success and challenges.

## Contact
For more information, please contact alkhali7@msu.edu
