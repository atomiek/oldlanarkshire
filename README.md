Old Lanarkshire is a student-facing web portal designed to be ran by a college/university.  For the project, a fictional College was created; “Old Lanarkshire”. It provides quick access to college resources and features an integrated AI chatbot named Roko — built and deployed entirely on local hardware, requiring no external AI API subscription, also reducing potential cyber security concerns, Old Lanarkshire should be as secure as the network it is deployed on (or cloud solution).
The OldLanarkshire chat-bot can be fed various information documents, since a fictional college was created for the chatbot generic educational information was fed to, and in a production environment it can be fed college information such as, unit specs, courses that are able to be applied to, etc.

This project is designed to be ran on a ubuntu machine, hyper-v or wsl2 with ubuntu can also run this project.

---------- Installation ----------
Download the installation files
cd into the root (where this README is)
sudo chmod +x setup.sh
./setup.sh

---------- Running the webserver ----------
sh run.sh 
then choose the model you want to run
you can then go to 127.0.0.1:8080