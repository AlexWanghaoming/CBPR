import os

_curr_directory = os.path.dirname(os.path.abspath(__file__))

BCP_MODELS = {'cramped_room': os.path.join(_curr_directory, 'bcp/bcp_cramped_room-seed1.pth'),
             'marshmallow_experiment': os.path.join(_curr_directory, 'bcp/bcp_marshmallow_experiment-seed1.pth'),
              'asymmetric_advantages': os.path.join(_curr_directory, 'bcp/bcp_asymmetric_advantages-seed1.pth')
              }

BC_MODELS = {'cramped_room': os.path.join(_curr_directory, 'bc/BC_cramped_room.pth'),
             'marshmallow_experiment': os.path.join(_curr_directory, 'bc/BC_marshmallow_experiment.pth'),
             'asymmetric_advantages': os.path.join(_curr_directory, 'bc/BC_asymmetric_advantages.pth')
             }

HP_MODELS = {'cramped_room': os.path.join(_curr_directory, 'bc/HP_simple.pth'),
             'marshmallow_experiment': os.path.join(_curr_directory, 'bc/HP_marshmallow_experiment.pth'),
             'asymmetric_advantages': os.path.join(_curr_directory, 'bc/HP_asymmetric_advantages.pth')
             }

META_TASK_MODELS = {'cramped_room':[os.path.join(_curr_directory,'bc/BC_cramped_room_(0.0, 0.0, 0.0, 0.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_cramped_room_(0.0, 0.0, 1.0, 0.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_cramped_room_(0.0, 1.0, 0.0, 0.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_cramped_room_(1.0, 0.0, 0.0, 0.0).pth')
                                    ],
                    'marshmallow_experiment':[os.path.join(_curr_directory,'bc/BC_marshmallow_experiment_(0.0, 0.0, 0.0, 0.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_marshmallow_experiment_(0.0, 0.0, 0.0, 1.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_marshmallow_experiment_(0.0, 0.0, 1.0, 0.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_marshmallow_experiment_(0.0, 1.0, 0.0, 0.0).pth'),
                                    os.path.join(_curr_directory,'bc/BC_marshmallow_experiment_(1.0, 0.0, 0.0, 0.0).pth')
                                    ],
                    'asymmetric_advantages': [
                                    os.path.join(_curr_directory, 'bc/BC_asymmetric_advantages_(0.0, 0.0, 0.0, 0.0).pth'),
                                    os.path.join(_curr_directory, 'bc/BC_asymmetric_advantages_(0.0, 0.0, 1.0, 0.0).pth'),
                                    os.path.join(_curr_directory, 'bc/BC_asymmetric_advantages_(0.0, 1.0, 0.0, 0.0).pth'),
                                    os.path.join(_curr_directory, 'bc/BC_asymmetric_advantages_(1.0, 0.0, 0.0, 0.0).pth'),
                                    ],
                    }

MTP_MODELS = {'cramped_room':
                  # [os.path.join(_curr_directory,'mtp/mtp_cramped_room-(0.0, 0.0, 0.0, 0.0)-seed42-gg.pth'),
                  #              os.path.join(_curr_directory,'mtp/mtp_cramped_room-(0.0, 0.0, 1.0, 0.0)-seed42-gg.pth'),
                  #              os.path.join(_curr_directory,'mtp/mtp_cramped_room-(0.0, 1.0, 0.0, 0.0)-seed42-gg.pth'),
                  #              os.path.join(_curr_directory,'mtp/mtp_cramped_room-(1.0, 0.0, 0.0, 0.0)-seed42-gg.pth'),
                  #              ],
              [os.path.join(_curr_directory, 'mtp/mtp_cramped_room-place_onion_in_pot-seed1.pth'),
                           os.path.join(_curr_directory,'mtp/mtp_cramped_room-deliver_soup-seed1.pth'),
                           os.path.join(_curr_directory,'mtp/mtp_cramped_room-place_onion_and_deliver_soup-seed1.pth'),
                           os.path.join(_curr_directory,'mtp/mtp_cramped_room-random-seed1.pth'),
                           ],
              'marshmallow_experiment':
                  [os.path.join(_curr_directory,'mtp/mtp_marshmallow_experiment-(0.0, 0.0, 0.0, 0.0)-seed42-gg.pth'),
                                           os.path.join(_curr_directory,'mtp/mtp_marshmallow_experiment-(0.0, 0.0, 0.0, 1.0)-seed42-gg.pth'),
                                           os.path.join(_curr_directory,'mtp/mtp_marshmallow_experiment-(0.0, 0.0, 1.0, 0.0)-seed42-gg.pth'),
                                           os.path.join(_curr_directory,'mtp/mtp_marshmallow_experiment-(0.0, 1.0, 0.0, 0.0)-seed42-gg.pth'),
                                           os.path.join(_curr_directory,'mtp/mtp_marshmallow_experiment-(1.0, 0.0, 0.0, 0.0)-seed42-gg.pth')
                                           ]
                  }


GP_MODELS = {'cramped_room': [os.path.join(_curr_directory,'gp/gp_cramped_room_(0.0, 0.0, 0.0, 0.0)_s_prime_r.pth'),
                                   os.path.join(_curr_directory,'gp/gp_cramped_room_(0.0, 0.0, 1.0, 0.0)_s_prime_r.pth'),
                                   os.path.join(_curr_directory,'gp/gp_cramped_room_(0.0, 1.0, 0.0, 0.0)_s_prime_r.pth'),
                                    os.path.join(_curr_directory,'gp/gp_cramped_room_(1.0, 0.0, 0.0, 0.0)_s_prime_r.pth'),
                                   ]
             }

NN_MODELS = {'cramped_room':
                 # [os.path.join(_curr_directory,'NN/NN_cramped_room_(0.0, 0.0, 0.0, 0.0)_s_prime_r.pth'),
                 #                   os.path.join(_curr_directory,'NN/NN_cramped_room_(0.0, 0.0, 1.0, 0.0)_s_prime_r.pth'),
                 #                   os.path.join(_curr_directory,'NN/NN_cramped_room_(0.0, 1.0, 0.0, 0.0)_s_prime_r.pth'),
                 #                    os.path.join(_curr_directory,'NN/NN_cramped_room_(1.0, 0.0, 0.0, 0.0)_s_prime_r.pth'),
                 #                   ],
             [os.path.join(_curr_directory, 'NN/NN_cramped_room_place_onion_in_pot_s_prime_r.pth'),
              os.path.join(_curr_directory, 'NN/NN_cramped_room_deliver_soup_s_prime_r.pth'),
              os.path.join(_curr_directory, 'NN/NN_cramped_room_place_onion_and_deliver_soup_s_prime_r.pth'),
              os.path.join(_curr_directory, 'NN/NN_cramped_room_random_s_prime_r.pth')],
            'marshmallow_experiment': [os.path.join(_curr_directory,'NN/NN_marshmallow_experiment_(0.0, 0.0, 0.0, 0.0)_s_prime_r.pth'),
                                       os.path.join(_curr_directory,'NN/NN_marshmallow_experiment_(0.0, 0.0, 0.0, 1.0)_s_prime_r.pth'),
                                       os.path.join(_curr_directory,'NN/NN_marshmallow_experiment_(0.0, 0.0, 1.0, 0.0)_s_prime_r.pth'),
                                        os.path.join(_curr_directory,'NN/NN_marshmallow_experiment_(0.0, 1.0, 0.0, 0.0)_s_prime_r.pth'),
                                        os.path.join(_curr_directory,'NN/NN_marshmallow_experiment_(1.0, 0.0, 0.0, 0.0)_s_prime_r.pth'),
                                               ],
             }






