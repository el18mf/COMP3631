# COMP3631
3rd Year Group Project for Intelligent Systems and Robotics module COMP3631

Report heavily embellished, after having to cover the entirety of the projects scope myself due to my team members having prioritsed their dissertations and producing no code nor research into their respective areas, resulting in my own dissertation suffering as I learnt and did my best over 10 days to create the resultant code and report by myself. 

The original scope breakdown was:
- Circle Detection: Mathew
- AI Face Detection: one member of the team was creating a face detecting software using machine learning for their dissertation, and offered to implement said software on our robot
- Autonomous Movement: Other two team members

Luckily my code and ideas for the circle detection were able to be used with repurposed for use with the face detection in a roundabout way, by forgoing the taught materials (basic face detection) I was able to use the Hu Moments to detect a range of colours associated with each cluedo character, and once detected the robot used stored values it obtained by calculating the Hu Moments of stored images of each character and comparing them to the detected Hu Moments, and would produce the closest matching character with a percentage of how close they matched. 
