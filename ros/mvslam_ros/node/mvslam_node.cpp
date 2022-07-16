#include "mvslam_node.h"

using namespace navw;


void test_chatter_cb(const std_msgs::String::ConstPtr& msg)
{
  string s = "Chatter receive:"+msg->data;
  ut::log::inf(s);
}

//-------------
// run_node()
//-------------
void run_node()
{



  //---- test chatter
  ros::Subscriber subt = nh_.subscribe("chatter", 1000, test_chatter_cb);


  //---- ROS mainloop
  ros::spin();


}
//------------
// main
//------------

int main(int argc, char **argv)
{
  using namespace std;
  ros::init(argc, argv, "mvslam_node");

  run_node();
  return 0;
}

