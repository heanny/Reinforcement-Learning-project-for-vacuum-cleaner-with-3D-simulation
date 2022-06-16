using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.Rendering;
using UnityEngine.UI;

public class CleanerAgent : Agent
{
    //define all parameter need to use
    int trashNum;
    Rigidbody rBody;
    int battery;
    System.Timers.Timer t;
    public Text txt;

    void Start () {
        rBody = GetComponent<Rigidbody>();
        trashNum = 0;
        battery = 100;
        txt.text = "Battery: " + battery.ToString() + "%";
        t = new System.Timers.Timer(50000);
        t.Elapsed += new System.Timers.ElapsedEventHandler(battery_decrese);
        t.AutoReset = true;
        t.Enabled = true;
    }


    // define start and end condition
    public override void OnEpisodeBegin()
    {
       // If the Agent fell, zero its momentum
        if (trashNum < 38)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
        }

    }

    // collect observation
    public override void CollectObservations(VectorSensor sensor)
    {
        // Target and Agent positions
        // sensor.AddObservation(this.transform.localPosition);

        // Agent velocity
        // sensor.AddObservation(rBody.velocity.x);
        // sensor.AddObservation(rBody.velocity.z);
        sensor.AddObservation(battery);
    }

    // define the action and movement
    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;
        var action = act[0];

        switch (action)
        {
            case 1:
                dirToGo = transform.forward * 1f;
                break;
            case 2:
                dirToGo = transform.forward * -1f;
                break;
            case 3:
                rotateDir = transform.up * 1f;

                break;
            case 4:
                rotateDir = transform.up * -1f;

                break;
        }
        //transform.Rotate(rotateDir, Space.Self)
        rBody.AddTorque(rotateDir, ForceMode.VelocityChange);
        rBody.AddForce(dirToGo, ForceMode.VelocityChange);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        MoveAgent(actionBuffers.DiscreteActions);
        if (trashNum == 38 || this.transform.localPosition.y < -2.5f)
        {
            t.Enabled = false;
            EndEpisode();
        }
        if (battery == 0)
        {
            battery = 100;
            SetReward(-10f);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 3;
        }
        else if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 4;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
    }

    //  define reward
    void OnCollisionEnter (Collision collisionInfo)
    {
        if (collisionInfo.collider.tag == "Obstacle")
        {
            SetReward(-1.0f);
        }
        if (collisionInfo.collider.tag == "Trash")
        {
            SetReward(10.0f);
            trashNum += 1;
            collisionInfo.gameObject.transform.localPosition = new Vector3(0, 100.0f, 0);
        }
        if (collisionInfo.collider.tag == "DoorTrash")
        {
            SetReward(15.0f);
            trashNum += 1;
            collisionInfo.gameObject.transform.localPosition = new Vector3(0, 100.0f, 0);
        }
        if (collisionInfo.collider.tag == "Charge")
        {
            if (battery <= 30){
                SetReward(20.0f);
            }
            battery = 100;
        }
    }

    // decrese of battery
    public void battery_decrese(object source, System.Timers.ElapsedEventArgs e)   
     {   
        battery -= 1;
        if (battery == 0)
        {
            battery = 100;
            SetReward(-10f);
        }
     }

     void Update()
     {
        
        txt.text = "Battery: " + battery.ToString() + "%";
        if (trashNum == 38 || this.transform.localPosition.y < -2.5f)
        {
            t.Enabled = false;
            EndEpisode();
        }
        if (battery == 0)
        {
            battery = 100;
            SetReward(-10f);
        }
    }
}