#include <opencv2/face.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <string>
#include <vector>

//prototypes in main, definitions here

//to normalize vector (used extensively)
static cv::Mat norm_0_255(cv::InputArray _src){
    cv::Mat src = _src.getMat();
    cv::Mat dst;
    switch(src.channels()) {
        case 1:
            cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}

//to read in from our database list file which is a csv file
static void read_csv(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels, char seperator = ';'){
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file){
        std::cout << "No valid input file was given. \n" <<
                     "Please check the given filename.\n";
    }
    std::string line, path, classlabel;
    while (getline(file,line)){
        std::stringstream liness(line);
        getline(liness, path, seperator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty()){
            cv::Mat m = cv::imread(path, 1);
            cv::Mat m2;
            if (m.channels() > 1){
            //if any input picture has more than 1 color channel we need to convert it to grayscale
            cvtColor(m,m2,CV_BGR2GRAY);
            }
            images.push_back(m2);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

std::string outputpathings(){
    std::string outputdir = ".";
    //std::string csv_in
    char dirspecifychoice;
    int flagy = 1;
    std::cout << "First, enter the directory where you wish to save the written image files.\n";
    std::cout << "By default, the directory is simply the current directory the program is being run in.\n";
    //please fix output folder to be able to be user-selectable
backloop:
    while (flagy == 1){
    std::cout << "Would you like to specify a custom directory? (Y\\N):" << std::endl;
    std::cin >> dirspecifychoice;
    if (dirspecifychoice == 'y' || dirspecifychoice == 'Y'){
        goto select;
    }
    else if (dirspecifychoice == 'n' || dirspecifychoice == 'N'){
        std::cout << "Ok. Using default directory (current one)\n";
        //flagy = 0;
        goto defaultdir;
    }
    else {
        std::cout << "Error! Incorrect input!\n";
        goto backloop;
    }
    }
select:
    std::cout << "Enter directory: ";
    std::cin >> outputdir;
defaultdir:
    return outputdir;
}

std::string csvpathings(){
    std::string csv_in;
    std::cout << "Enter file location for csv file containing face db\n";
    std::cout << "Example: 'att_faces/at.txt'" << "\n";
    std::cin >> csv_in;
    return csv_in;
}

//main function for fisherfaces; comments for eigenfaces function explain many of the function calls used here
int fisherfaces(){
    std::string outputdir = outputpathings();
    std::string csv_in = csvpathings();
    std::vector<cv::Mat> fisherimages;
    std::vector<int> fisherlabels;
    std::cout << "Checking for errors..." << "\n";
    try {
        read_csv(csv_in, fisherimages, fisherlabels);
    }
    catch (cv::Exception& e){
        std::cerr << "Error opening the file \"" << csv_in << "\". Reason: " << e.msg <<  "\n";
        return -1;
    }
    if (fisherimages.size() <= 1){
        std::cout << "This needs at least 2 images to work. Error. Exiting\n";
        return -1;
    }
    std::cout << "No errors detected. \n";
    std::cout << "Running through calculations..." << "\n";
    int height = fisherimages[0].rows;
    cv::Mat fisherTestingSample = fisherimages[fisherimages.size()-1];
    int fisherTestingLabel = fisherlabels[fisherlabels.size() - 1];
    fisherimages.pop_back();
    fisherlabels.pop_back();
    cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::createFisherFaceRecognizer();
    model->train(fisherimages, fisherlabels);
    int predictedLabel = model->predict(fisherTestingSample);
    std::string result_message = cv::format("Predicted class = %d / Actual class = %d.", predictedLabel, fisherTestingLabel);
    std::cout << result_message << "\n";
    cv::Mat eigenvalues = model->getEigenValues();
    cv::Mat W = model->getEigenVectors();
    cv::Mat mean = model->getMean();
    //if (argc == 2){
    imwrite(cv::format("%s/mean.png", outputdir.c_str()), norm_0_255(mean.reshape(1, fisherimages[0].rows)));

    for (int i = 0; i < cv::min(10, W.cols); i++) {
        std::string msg = cv::format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        std::cout << msg << "\n";
        cv::Mat ev = W.col(i).clone();
        cv::Mat grayscale = norm_0_255(ev.reshape(1, height));
        cv::Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, cv::COLORMAP_BONE);
            imwrite(cv::format("%s/fisherface_%d.png", outputdir.c_str(), i), norm_0_255(cgrayscale));
    }

    for(int num_component = 0; num_component < cv::min(16, W.cols); num_component++){
        cv::Mat ev = W.col(num_component);
        cv::Mat fisherprojection = cv::LDA::subspaceProject(ev, mean, fisherimages[0].reshape(1,1));
        cv::Mat fisherreconstruction = cv::LDA::subspaceReconstruct(ev, mean, fisherprojection);
        //normalize result
        fisherreconstruction = norm_0_255(fisherreconstruction.reshape(1, fisherimages[0].rows));
            imwrite(cv::format("%s/fisherface_reconstruction_%d.png", outputdir.c_str(), num_component), fisherreconstruction);
    }
    return 0;
}

int lbp(){
    std::cout << "LBPH cannot collect the features required for facial reconstruction\n";
    std::cout << "However, it can still output the data of the images the model is being trained to read.\n";
    std::string outputdir = outputpathings();
    std::string csv_in = csvpathings();
    std::vector<cv::Mat> lbp_images;
    std::vector<int> lbp_labels;
    std::cout << "Checking for Errors...\n";
    try {
        read_csv(csv_in, lbp_images, lbp_labels);
    }
    catch (cv::Exception& e){
        std::cerr << "Error opening file \"" << csv_in << "\". Reason: " << e.msg << "\n";
        return -1;
    }
    if (lbp_images.size() <= 1){
        std::cout << "Error: This needs 2 or more images to work. Exiting...\n";
        return -1;
    }
    std::cout << "No errors detected.\n";
    std::cout << "Running through calculations...\n";
    //int height = lbp_images[0].rows;
    cv::Mat lbpTestingSample = lbp_images[lbp_images.size() - 1];
    int lbpTestingLabel = lbp_labels[lbp_labels.size() - 1];
    lbp_images.pop_back();
    lbp_labels.pop_back();
    cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::createLBPHFaceRecognizer();
    model -> train(lbp_images,lbp_labels);
    //predict labels
    int predicted_lbp_label= model->predict(lbpTestingSample);
    std::string result_message = cv::format("Predicted class = %d / Actual class = %d.", predicted_lbp_label, lbpTestingLabel);
    std::cout << result_message << "\n";
    // We don't use a particular threshold because we use multiple sets of data. Best to set threshold to 0.
    model->setThreshold(0.0);
    predicted_lbp_label = model->predict(lbpTestingSample);
    //std::cout << "Predicted class = " << predicted_lbp_label << "\n";
    std::cout << "Model Information: " << "\n";
    std::string model_info = cv::format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
            model->getRadius(),
            model->getNeighbors(),
            model->getGridX(),
            model->getGridY(),
            model->getThreshold());
    std::cout << model_info;
    std::cout << "\n";
    std::vector<cv::Mat> histograms = model-> getHistograms();
    std::cout << "Size of histograms: " << histograms[0].total() << "\n";
    //need to make for loops for reconstruction
    //cv::imwrite(cv::format("%s/lbph_%d.png", outputdir.c_str(), i), norm_0_255(cgrayscale));
    //cv::imwrite(cv::format("%s/lbph_reconstruction_%d.png", outputdir.c_str(), num_components, lbph_reconstruction);

return 0;
}

int eigenfaces(){
    std::string outputdir = outputpathings();
    std::string csv_in = csvpathings();
    //vectors to hold the corresponding images and labels
    std::vector<cv::Mat> eigen_images;
    std::vector<int> eigen_labels;
    std::cout << "Checking for errors..." << "\n";
    try {
        read_csv(csv_in, eigen_images, eigen_labels);
    }
    catch (cv::Exception& e){
        std::cerr << "Error opening csv file \"" << csv_in << "\". Reason: " << e.msg << "\n";
        //exit with -1 error!
        return -1;
    }
    if (eigen_images.size() <= 1){
        std::cout << "This program requires 2 or more images. Exiting...\n";
        return -1;
    }
    std::cout << "No errors detected." << "\n";
    std::cout << "Running through calculations..." << "\n";
    //Basically we need height from first image because we need to resize the 
    //other images. Every image must be the same size.
    int height = eigen_images[0].rows;
    // The following lines get the last images from the dataset
    // and remove it from the vector.
    cv::Mat eigenTestingSample = eigen_images[eigen_images.size()-1];
    int eigenTestingLabel = eigen_labels[eigen_labels.size() - 1];
    eigen_images.pop_back();
    eigen_labels.pop_back();
    //create eigenfaces model for face recognition then
    //train it with images and labels of the csv file
    //we use full pca (no arguments to createEigenFaceRecognizer)
    cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::createEigenFaceRecognizer();
    model->train(eigen_images, eigen_labels);
    //predict label of given test image
    int predictedLabel = model->predict(eigenTestingSample);
    std::string result_message = cv::format("Predicted class = %d / Actual class = %d.", predictedLabel, eigenTestingLabel);
    
    std::cout << result_message << "\n";
    cv::Mat eigenvalues = model->getEigenValues();
    cv::Mat W = model->getEigenVectors();
    cv::Mat mean = model->getMean();
    cv::imwrite(cv::format("%s/mean.png", outputdir.c_str()), norm_0_255(mean.reshape(1, eigen_images[0].rows)));
    
    for (int i = 0; i < cv::min(10, W.cols); i++) {
        std::string msg = cv::format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        std::cout << msg << "\n";
        cv::Mat ev = W.col(i).clone();
        cv::Mat grayscale = norm_0_255(ev.reshape(1, height));
        cv::Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, cv::COLORMAP_JET);
        cv::imwrite(cv::format("%s/eigenface_%d.png", outputdir.c_str(), i), norm_0_255(cgrayscale));
    }

    for(int num_components = cv::min(W.cols, 10); num_components < cv::min(W.cols, 300); num_components+=15){
        cv::Mat evs = cv::Mat(W, cv::Range::all(), cv::Range(0, num_components));
        cv::Mat eigenfacesprojection = cv::LDA::subspaceProject(evs, mean, eigen_images[0].reshape(1,1));
        cv::Mat eigenfacesreconstruction = cv::LDA::subspaceReconstruct(evs, mean, eigenfacesprojection);
        eigenfacesreconstruction = norm_0_255(eigenfacesreconstruction.reshape(1, eigen_images[0].rows));
        cv::imwrite(cv::format("%s/eigenface_reconstruction_%d.png", outputdir.c_str(), num_components), eigenfacesreconstruction);
    }

return 0;
}

//program startup text
int intro(int argc){
    int count = (4+argc-argc);
    for (int i = 0; i < count; i++){
        std::cout << "**************************************************************" << "\n";
        }
    //intro();
    std::cout<<"Welcome to my Final Project on Facial Reconstruction!\n";
    std::cout<<"All code written by Christopher McFee\n";
    std::cout<<"This source code uses many functions from OpenCV documentation on BasicFaceRecognizer\n";
    std::cout<<"Please see project sources for more information\n";
    std::cout<<"http://docs.opencv.org/ref/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html\n";
    std::cout<<"http://docs.opencv.org/ref/master/dc/dd7/classcv_1_1face_1_1BasicFaceRecognizer.html\n";
    std::cout<<"Related Face Recognizers Are Also included in the OpenCV3 documentation.\n";
    std::cout<<"FaceRecognizer and BasicFaceRecognizer have changed quite a bit from OpenCV2.\n";
    std::cout<<"Thus, the old contrib library from OpenCV2 is incompatible with this code.\n";
    std::cout<<"Please see the OpenCV facerec.hpp file for more information\n";
    std::cout << "This code makes heavy use of the OpenCV contrib libraries.\n";
    //depends on OpenCV 3.1, contrib library, and cmake (OSX, Windows, and GNU/Linux compatible)
    //tested under Mac OSX, GNU/Linux, and Windows with Cygwin
    std::cout << "I'd love to thank the original authors and contributors for all of their hard work!\n";
    std::cout << "This project would not be possible without them!\n";
    for (int i = 0; i < count; i++){
        std::cout << "*************************************************************" << "\n";
    }
    std::cout << "This is a facial recognition program that uses \n";
    std::cout << "images from two databases, the AT&T face database and the Sheffield University database.\n";
    std::cout << "This program uses a basic text menu interface with iostream from the STL.\n";
    for (int i = 0; i < count; i++){
        std::cout << "*************************************************************" << "\n";
    }
return 0;
}

//display menu function: simple for now
void displaymenu(){
    std::cout << "Facial Recognition can use various algorithms." << "\n";
    std::cout << "Select one. (0: Short Explanation of Each)" << "\n";
    std::cout << "1: Eigenfaces" << "\n";
    std::cout << "2: Fisherfaces" << "\n";
    std::cout << "3: Local Binary Patterns Histograms" << "\n";
    std::cout << "(Use numkeys to select algorithm)" << "\n";
}

//takes an input from user
int usermenuselection(){
    int selection;
    std::cout << "Selection: ";
    std::cin >> selection;
    while (std::cin.fail()){
        std::cout << "Error. Input not valid type" << "\n";
        std::cin.clear();
        std::cin.ignore(256,'\n');
        std::cout << "Selection: ";
        std::cin >> selection;
    }
    while ((selection != 0) &&
    (selection != 1) && (selection !=2) && (selection != 3)){
        std::cout << "Selection " << selection << " is invalid. " << "\n";
        std::cout << "Input Selection:" << "\n";
        std::cout << "Selection: ";
        std::cin >> selection;
    }
    return selection;
}

//print user selection
int printselection(int selection){
    if (selection == 0){
        std::cout << "Which algorithm to explain? (1,2,3):" << "\n";
        return 0;
    }
    else if (selection == 1){
        std::cout << "Eigenfaces" << "\n";
        std::cout << "..." << "\n";
    }
    else if (selection == 2){
        std::cout << "Fisherfaces" << "\n";
        std::cout << "..." << "\n";
    }
    else if (selection == 3){
        std::cout << "Local Binary Patterns" << "\n";
        std::cout << "..." << "\n";
    }
    else{
        std::cout << "ERROR: Input was not properly sanitized!" << "\n";
        return -1;
    }
    return 1;
}

//need to finish this function; all algorithms need brief (not super detailed) descriptions here:
int algorithmdescriptions(int selection){
    //this needs to be in a loop
    //also needs serious formatting improvement
    if (selection == 1){
    std::cout << "1. Eigenfaces: " << "\n";
    std::cout << "Eigenfaces is a holistic approach to facial recognition.\n";
    std::cout << "A facial image is a point from a high-dimensional image space and a lower-dimensional representation is found, where classificiation becomes easy." << "\n";
       std::cout << "Lower dimensional subspace is found with Principal Component Analysis (PCA), which identifies the axes with maximum variance. This is not very useful when you have variance from multiple external sources (ie light)\n" <<
        "In some cases, the axes with maximum variance do not necessarily contain any discriminiative information at all, so a classification becomes impossible." << "\n" <<
        "So, a class-specific projection with a Linear Discrimininant Analysis (LDA) was applied to facial recognition. (also applies to fisher faces.)" << "\n" <<
        "Basic Idea is to minimize variance within a class, while maximizing variance between classes at same time." << "\n" <<
        "To do this we use LDA:\nWhat is LDA?\n" <<
        "LDA is a generalization of Ronald Fisher's linear discriminant:\n" <<
        "It looks for linear combinations of variables which best explain the data\n" <<
        //"z=B1X1+B2X2+...+BdXd\n" <<
        "LDA is based upon the concept of searching for a linear combination of variables that best seperates 2 classes. To capture notion of seperability, Fisher defined an algorithm\n" <<
        "One way of assessing the effectiveness of the discrimination is to calculate the Mahalanobis distance between 2 groups. A distance greater than 3 means that in 2 averages differ by more than 3 standard deviations. It means that the overlap (probability of misclassification) is quite small.\n" <<
        "PCA is used to turn a set of possibly correlated variables into a smaller set of uncorrelated variables." <<"\n";
    }
    else if (selection == 2){
        std::cout << "2. Fisherfaces" << "\n";
        std::cout << "The Principal Component Analysis (PCA), which is the core of the Eigenfaces method, finds a linear combination of features that maximizes the total variance in data. While this is clearly a powerful way to represent data, it doesn’t consider any classes and so a lot of discriminative information may be lost when throwing components away.\n" << 
"Imagine a situation where the variance in your data is generated by an external source, let it be the light. The components identified by a PCA do not necessarily contain any discriminative information at all, so the projected samples are smeared together and a classification becomes impossible\n" << 
" (see http://www.bytefish.de/wiki/pca_lda_with_gnu_octave for an example).\n" << 
"The Linear Discriminant Analysis performs a class-specific dimensionality reduction and was invented by the great statistician Sir R. A. Fisher. He successfully used it for classifying flowers in his 1936 paper The use of multiple measurements in taxonomic problems [Fisher36]. In order to find the combination of features that separates best between classes the Linear Discriminant Analysis maximizes the ratio of between-classes to within-classes scatter, instead of maximizing the overall scatter.\n" << 
"The idea is simple: same classes should cluster tightly together, while different classes are as far away as possible from each other in the lower-dimensional representation. This was also recognized by Belhumeur, Hespanha and Kriegman and so they applied a Discriminant Analysis to face recognition" << "\n";
    }

    else if (selection == 3){
        std::cout << "3. Local Binary Patterns Histograms" << "\n";
        std::cout << "The idea is to not look at the whole image as a high-dimensional vector, but describe only local features of an object.\n" << 
"The features you extract this way will have a low-dimensionality implicitly. A fine idea! But you’ll soon observe the image representation we are given doesn’t only suffer from illumination variations. Think of things like scale, translation or rotation in images - your local description has to be at least a bit robust against those things. Just like SIFT, the Local Binary Patterns methodology has its roots in 2D texture analysis.\n" << 
"The basic idea of Local Binary Patterns is to summarize the local structure in an image by comparing each pixel with its neighborhood. Take a pixel as center and threshold its neighbors against. If the intensity of the center pixel is greater-equal its neighbor, then denote it with 1 and 0 if not. You’ll end up with a binary number for each pixel, just like 11001111. So with 8 surrounding pixels you’ll end up with 2^8 possible combinations, called Local Binary Patterns or sometimes referred to as LBP codes.\n" << 
"This description enables you to capture very fine grained details in images."<< "\n";
    }

    else{
        return 9;
    }
return 0;
}

int selectalgode(int qselection, int selection){
    int qinput;
    int localselection = selection;
    localselection = 0;
    localselection = qselection;
    std::cout << "Select one.(1,2,3)" << "\n";
    std::cout << "1: Eigenfaces" << "\n";
    std::cout << "2: Fisherfaces" << "\n";
    std::cout << "3: Local Binary Patterns Histograms" << "\n";
    std::cout << "(Use numkeys to select algorithm)" << "\n";
    std::cin >> qinput;
    std::cout << "You entered: " << qinput << "\n";
    std::cout << "\n";
    //std::cout << ":" << "\n";
    //std::cin >> qselectioni;
    return qinput;
}

