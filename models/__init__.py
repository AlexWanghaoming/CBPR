import os

_curr_directory = os.path.dirname(os.path.abspath(__file__))

BCP_MODELS = {'cramped_room': os.path.join(_curr_directory, 'bcp/bcp_cramped_room-seed42.pth'),
             'marshmallow_experiment': os.path.join(_curr_directory, 'bcp/bcp_marshmallow_experiment-seed42.pth')}

BC_MODELS = {'cramped_room': os.path.join(_curr_directory, 'bc/BC_simple.pth'),
             'marshmallow_experiment': os.path.join(_curr_directory, 'bc/BC_marshmallow_experiment.pth')}

META_TASK_MODELS = {'cramped_room':[os.path.join(_curr_directory,'bc/BC_simple_(0.0, 0.0, 0.0, 0.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_simple_(0.0, 0.0, 1.0, 0.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_simple_(0.0, 1.0, 0.0, 0.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_simple_(1.0, 0.0, 0.0, 0.0).pth')
                                    ],
                    'marshmallow_experiment':[os.path.join(_curr_directory,'bc/BC_marshmallow_experiment_(0.0, 0.0, 0.0, 0.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_marshmallow_experiment_(0.0, 0.0, 0.0, 1.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_marshmallow_experiment_(0.0, 0.0, 1.0, 0.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_marshmallow_experiment_(0.0, 1.0, 0.0, 0.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_marshmallow_experiment_(1.0, 0.0, 0.0, 0.0).pth')
                                    ],
                    }

MTP_MODELS = {'cramped_room': [os.path.join(_curr_directory,'mtp/mtp_cramped_room-(0.0, 0.0, 0.0, 0.0)-seed42.pth'),
                                   os.path.join(_curr_directory,'mtp/mtp_cramped_room-(0.0, 0.0, 1.0, 0.0)-seed42.pth'),
                                   os.path.join(_curr_directory,'mtp/mtp_cramped_room-(0.0, 1.0, 0.0, 0.0)-seed42.pth'),
                                   os.path.join(_curr_directory,'mtp/mtp_cramped_room-(1.0, 0.0, 0.0, 0.0)-seed42.pth'),
                                   ],
                'marshmallow_experiment': [os.path.join(_curr_directory,'mtp/mtp_marshmallow_experiment-(0.0, 0.0, 0.0, 0.0)-seed42.pth'),
                                                   os.path.join(_curr_directory,'mtp/mtp_marshmallow_experiment-(0.0, 0.0, 0.0, 1.0)-seed42.pth'),
                                                   os.path.join(_curr_directory,'mtp/mtp_marshmallow_experiment-(0.0, 0.0, 1.0, 0.0)-seed42.pth'),
                                                   os.path.join(_curr_directory,'mtp/mtp_marshmallow_experiment-(0.0, 1.0, 0.0, 0.0)-seed42.pth'),
                                                   os.path.join(_curr_directory,'mtp/mtp_marshmallow_experiment-(1.0, 0.0, 0.0, 0.0)-seed42.pth')
                                                   ]
                  }