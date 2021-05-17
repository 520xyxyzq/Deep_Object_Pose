#! /usr/bin/env python

'''
evaluate.py: script to evaluate dope cards detector
             including mis-classification rate & pose estimation accuracy
'''

__author__ = "Ziqi Lu, Violet Killy"
__email__ = 'ziqilu@mit.edu, violetk@mit.edu'
__copyright__ = "Copyright 2021 The Ambitious Folks of the MRG"

import rospy
import cv2
import numpy as np

from apriltag_ros.msg import AprilTagDetectionArray # for card/tag poses
from vision_msgs.msg import Detection3DArray # for dope poses

from scipy.spatial.transform import Rotation as Rot

class CardsEvaluate(object):

    def __init__(self):

        self.tag_size, self.card_size = 0.05, [0.057, 0.089]
        # Size of the interior square bounded by the tag corners
        self.square_size = 0.1
        # Tag ID (in /tag_detections msg published by apriltag_ros)
        self.tags = np.array([0, 1, 2, 3])
        # Relative translations of test card CENTER wrt tags 
        self.card2tags = np.array([
            [ (self.tag_size+self.square_size)/2, 
              (self.tag_size+self.square_size)/2, 0],
            [-(self.tag_size+self.square_size)/2, 
              (self.tag_size+self.square_size)/2, 0],
            [ (self.tag_size+self.square_size)/2,
             -(self.tag_size+self.square_size)/2, 0],
            [-(self.tag_size+self.square_size)/2,
             -(self.tag_size+self.square_size)/2, 0]
            ])
        # Subscribe to tags' id-poses 
        rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.tag_det_cb)
        # Subscriber for feature based card pose estimator
        rospy.Subscriber('/card_detections', AprilTagDetectionArray, self.card_det_cb)
        #Subscriber for dope poses
        rospy.Subscriber('/dope/detected_objects', Detection3DArray, self.dope_det_cb)

        # Counters:
        self.inlier, self.outlier, self.dets = 0, 0, 0
        # Translation & rotation distances
        self.dist_rot, self.dist_trans = [], []

        self.ctrl_c = False
        rospy.on_shutdown(self.shutdownhook)

    def shutdownhook(self):
        if not self.dets:
            rospy.logwarn("No card detection!")
        else:
            rospy.loginfo("Number of detections: %d", self.dets)
            rospy.loginfo("True positive rate: %.3f", 
                self.inlier/np.double(self.dets))
            rospy.loginfo("False positive rate: %.3f",
                self.outlier/np.double(self.dets))
            rospy.loginfo("Average translation distance: %.3f", 
                sum(self.dist_trans)/(1e-20+len(self.dist_trans)))
            rospy.loginfo("Average rotation distance: %.3f", 
                sum(self.dist_rot)/(1e-20+len(self.dist_rot)))            
        self.ctrl_c = True

    def tag_det_cb(self, msg):
        '''
        Callback function for /tag_detections msg (from apriltag_ros)  
        Input: 
            msg(AprilTagDetectionArray msg)
        '''
        dets = msg.detections # detections msg
        tag_trans = np.zeros((len(dets),3))
        tag_quats = np.zeros((len(dets),4))
        for ii in range(len(dets)):
            # Relative orientation of tag
            tag_quats[ii,:] = np.array([dets[ii].pose.pose.pose.orientation.x,
                                        dets[ii].pose.pose.pose.orientation.y,
                                        dets[ii].pose.pose.pose.orientation.z,
                                        dets[ii].pose.pose.pose.orientation.w])
            card2tag = self.card2tags[self.tags==dets[ii].id[0]].ravel()
            # Relative position of tag center
            tag_trans[ii,:] = np.array([dets[ii].pose.pose.pose.position.x,
                                        dets[ii].pose.pose.pose.position.y,
                                        dets[ii].pose.pose.pose.position.z])\
                            + Rot.from_quat(tag_quats[ii,:]).as_dcm()\
                            .dot(card2tag) 

        if dets:    
            self.tag_trans = np.mean(tag_trans, 0)
            # This is not rigorous but work as orientations are very similar
            # TODO(ziqi): make orientation avg rigorous
            tag_quat = np.mean(tag_quats, 0)
            # Transform tag's coordinate system to card's (180deg roll flip)
            tag_quat = self.quat_roll_flip(tag_quat)

            # If yaw angle is not within [-90, 90] deg, map the angle to within it   
            rpy = Rot.from_quat(tag_quat).as_euler("xyz", degrees=True)    
            if rpy[-1] > 90 or rpy[-1] < -90:
                self.tag_quat = self.quat_yaw_flip(tag_quat)
            else:
                self.tag_quat = tag_quat
        # Uncomment to debug
        #rospy.loginfo("Average tag translation: %s", str(self.tag_trans))       
        #rospy.loginfo("Average tag quaternion: %s", str(self.tag_quat))  

    def card_det_cb(self, msg):
        '''
        Callback function for /card_detections msg (feature based method)  
        Input: 
            msg(AprilTagDetectionArray msg)
        '''
        # Query tag pose first to minimize lag
        tag_trans, tag_quat = self.tag_trans, self.tag_quat
        # Increment #detections
        self.dets += 1
        # card detection
        dets = msg.detections
        # It's possible but rare to have 2 detections in the same frame
        # TODO(ziqi): deal with this situation
        if len(dets) > 2: rospy.logwarn("More than 1 card detected")
        elif len(dets) == 0:
            rospy.logdebug("No card detected")
        else:
            card_quat = np.array([dets[0].pose.pose.pose.orientation.x,
                                   dets[0].pose.pose.pose.orientation.y,
                                   dets[0].pose.pose.pose.orientation.z,
                                   dets[0].pose.pose.pose.orientation.w])
            # card coordinate from corner to center 
            card_trans = np.array([dets[0].pose.pose.pose.position.x,
                                   dets[0].pose.pose.pose.position.y,
                                   dets[0].pose.pose.pose.position.z])\
                        + Rot.from_quat(card_quat).as_dcm()\
                        .dot(np.array(self.card_size+[0.0])/2)

            # Compute the distance to ground truth
            dist_trans = np.linalg.norm(card_trans - tag_trans)
            dist_rot = self.computeRotDist(card_quat, tag_quat) 
            # Treat detection as outlier if relative translation > threshold
            # NOTE: we may need to tune the threshold 
            # or make more advanced criteria to correctly separate outliers
            if dist_trans > self.square_size:
                self.outlier += 1
            else:
                self.inlier += 1
                self.dist_trans.append(dist_trans)
                self.dist_rot.append(dist_rot)

    def dope_det_cb(self, msg):
        '''
        Callback function for /dope/detected_objects msg (dope published)  
        Input: 
            msg(Detection3DArray msg)
        '''
         
        # Query tag pose first to minimize lag
        tag_trans, tag_quat = self.tag_trans, self.tag_quat
        align = np.array([[0,1,0], [0,0,-1], [-1, 0, 0]])
        # Increment #detections
        self.dets += 1
        # dope detection
        dets = msg.detections
        if len(dets) > 1: rospy.logwarn("More than 1 card detected")
        elif len(dets) == 0:
            rospy.logdebug("No card detected")
        else:
            dope_quat =np.array([dets[0].results[0].pose.pose.orientation.x,
                                       dets[0].results[0].pose.pose.orientation.y,
                                       dets[0].results[0].pose.pose.orientation.z,
                                       dets[0].results[0].pose.pose.orientation.w])
            
            Rm = Rot.from_quat(dope_quat).as_dcm() 
            dope_quat = Rot.from_dcm(Rm.dot(align)).as_quat()
            
            # card coordinate from corner to center 
            dope_trans = np.array([dets[0].results[0].pose.pose.position.x,
                                       dets[0].results[0].pose.pose.position.y,
                                       dets[0].results[0].pose.pose.position.z])           
                    
            # Compute the distance to ground truth
            rpy = Rot.from_quat(dope_quat).as_euler("xyz", degrees=True)    
            if rpy[-1] > 90 or rpy[-1] < -90:
                dope_quat = self.quat_yaw_flip(dope_quat)      

            dist_trans = np.linalg.norm(dope_trans - tag_trans)
            dist_rot = self.computeRotDist(dope_quat, tag_quat) 
            
            rpy1 = Rot.from_quat(dope_quat).as_euler("xyz", degrees=True)[-1]
            rpy2 = Rot.from_quat(tag_quat).as_euler("xyz", degrees=True)[-1]
            #print(rpy1,rpy2 )
            if dist_rot>1.5 and abs(rpy1-rpy2)<5: 
                rospy.logwarn("%.4f>1.5"%dist_rot)
                print(dope_quat, tag_quat)
            # Treat detection as outlier if relative translation > threshold
            # NOTE: we may need to tune the threshold 
            # or make more advanced criteria to correctly separate outliers
            if dist_trans > self.square_size:
                self.outlier += 1
            else:
                self.inlier += 1
                self.dist_trans.append(dist_trans)
                self.dist_rot.append(dist_rot)

    def computeRotDist(self, quat1, quat2):
        '''
        Util function to compute the chordal distance between two quaterninons
        Input:
            quat1(4-array)
            quat2(4-array)
        Output:
            dist(double)
        '''
        assert len(quat1) == 4 & len(quat2) == 4
        rpy1 = Rot.from_quat(quat1).as_euler("xyz", degrees=True) 
        rpy2 = Rot.from_quat(quat2).as_euler("xyz", degrees=True) 
        if rpy1[-1]*rpy2[-1]<0 and abs(rpy1[-1]-90)<20 and (rpy2[-1]-90)<20:
            quat1 = self.quat_yaw_flip(quat1)
        rotmat1 = Rot.from_quat(quat1).as_dcm()
        rotmat2 = Rot.from_quat(quat2).as_dcm()
        return np.linalg.norm(rotmat1-rotmat2)

    def quat_roll_flip(self, quat_in):
        '''
        Util function to flip a quaternion's yaw angle by 180deg  
        Input:
            quat_in (4-array) 
        Output:
            quat_out (4-array)
        '''
        assert len(quat_in) == 4
        rotmat = Rot.from_quat(quat_in).as_dcm()
        trans_matrix = np.eye(3)
        trans_matrix[1,1], trans_matrix[2,2] = -1.0, -1.0
        return Rot.from_dcm(rotmat.dot(trans_matrix)).as_quat()        

    def quat_yaw_flip(self, quat_in):
        '''
        Util function to flip a quaternion's yaw angle by 180deg  
        Input:
            quat_in (4-array) 
        Output:
            quat_out (4-array)
        '''
        assert len(quat_in) == 4
        rotmat = Rot.from_quat(quat_in).as_dcm()
        trans_matrix = np.eye(3)
        trans_matrix[1,1], trans_matrix[0,0] = -1.0, -1.0
        return Rot.from_dcm(rotmat.dot(trans_matrix)).as_quat()

    def main(self):
        while not self.ctrl_c:
            continue

if __name__ == "__main__":
    rospy.init_node("cards_evaluate_node", anonymous = True)
    cards_evaluate = CardsEvaluate()
    try:
        cards_evaluate.main()
    except rospy.ROSInterruptException:
        pass

    