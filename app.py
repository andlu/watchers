# -*- coding: utf-8 -*-

import flask
import os
import pandas as pd
import cv_cam_facial_expression as tk
import Rec as recmod
import scipy
import warnings
import flask
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
video_id = ""
sub1 = ""


@app.route("/", methods=['GET', 'POST'])
def start():
    return render_template("homepage.html")


@app.route("/render", methods=['GET'])
def home():
    if request.method == 'GET':
        sub1 = request.args.get('subject')
        video_id = request.args.get('video_id')
        return render_template("new.html", state="NONE", wl="NONE", seek_time="NONE", sub=sub1, you_vid=video_id)


@app.route("/launch", methods=['GET', 'POST'])
def launch():
    if request.method == 'GET':
        sub1 = request.args.get('subject')
        video_id = request.args.get('video_id')
        tk.cam_run()
        return render_template("new.html", state="None", wl="NONE", seek_time="None", you_vid=video_id, sub=sub1)


@app.route("/rec", methods=['GET'])
def perform_rec():
    print("call for perform")
    if request.method == 'GET':
        sub1 = request.args.get('subject')
        vid_id = request.args.get('video_id')
        skt = request.args.get('seek_time')
        id = skt.find('.')
        skt = skt[:id]
        val = recmod.call_rec(sub1, vid_id, skt)
        # val = " ".join(val)
        return render_template("new.html", state="NONE", wl=val,
                               seek_time=skt, you_vid=vid_id, sub=sub1)


@app.route("/results", methods=['GET'])
def get_results():
    print("call for perform")
    if request.method == 'GET':
        score = request.args.get('score')
        return render_template("results.html", score=score)


if __name__ == "__main__":
    app.run(debug=True)
