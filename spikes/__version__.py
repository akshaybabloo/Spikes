#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2016 Akshay Raj Gollahalli

import json

version_json = '''
{
 "version": "0.2.9"
}
'''


def get_versions():
    return json.loads(version_json)
