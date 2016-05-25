#include "fn.h"
#include <string>

//function prototypes
static cv::Mat norm_0_255(cv::InputArray _src);
static void read_csv(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels, char seperator);
int fisherfaces();
int lbp();
int eigenfaces();
void displaymenu();
int usermenuselection();
int printselection();
int algorithmdescriptions(int);
int selectalgode(int, int);
int intro(int);
std::string outputpathings();
std::string csvpathings();

//main function
int main (int argc, char *argv[]){
    //display introduction text
    intro(argc);
    int qselection = 0;
    int main_flag = 1;
    std::string str1_1 = argv[3];
start:
    while (main_flag == 1){
    //run the menu function
    displaymenu();
    //input user selection and save; printmyselection is for a different function
    //which is related to usermenuselection
    int selection = usermenuselection();
    int printmyselection = printselection(selection);
    //if the selection is 0, run the explain function
    if (printmyselection == 0){
        std::cout << "You've opted for an explanation\n" <<
            "of the algorithms present.\n" << std::endl;
        qselection=selectalgode(qselection,selection);
        algorithmdescriptions(qselection);
    }
    //as long as printselection == 1, loop shall begin
    else if (printmyselection == -1){
        std::cout << "A serious error has been detected. Now exiting." << "\n";
        //change this to print debug info here
        return -1; //exit with error information
    }
    else if (printmyselection == 1){
        std::cout << "All set......" << "\n";
        for (int i = 4; i > 0; i--){
            std::cout << "**************************************************************" << "\n";
        }
    }
    qselection = selection;
    //here we decide what gets done
    if (qselection == 1){
        main_flag = 0;
        eigenfaces();
    }
    else if (qselection == 2){
        main_flag = 0;
        fisherfaces();
    }
    else if (qselection == 3){
        main_flag = 0;
        lbp();
    }
    else {
        main_flag = 1;
    }
    while (main_flag == 1){
    char yesno;
    std::cout << "Would you like to select an algorithm to use now? (Y\\N)\n";
    std::cin >> yesno;
    if (yesno == 'y' || yesno == 'Y'){
        std:: cout << "Going Back To Algorithm Selection..." << std::endl;
        goto start;
    }
    else if (yesno == 'n' || yesno == 'N'){
        std::cout << "Exiting..." << std::endl;
        main_flag = 0;
        goto stop;
    }
    else{
        std::cout << "Invalid entry!" << std::endl;
    }
    }
    }
stop:
    return 0;
}

