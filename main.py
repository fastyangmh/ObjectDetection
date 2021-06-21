# import
from src.project_parameters import ProjectParameters
from src.train import train
from src.predict import Predict

# def


def main(project_parameters):
    result = None
    if project_parameters.mode == 'train':
        result = train(project_parameters=project_parameters)
    elif project_parameters.mode == 'evaluate':
        pass
    elif project_parameters.mode == 'predict':
        result = Predict(project_parameters=project_parameters).get_result(
            data_path=project_parameters.data_path)
        print(('{},'*project_parameters.num_classes).format(*
                                                            project_parameters.classes.keys())[:-1])
        print(result)
    elif project_parameters.mode == 'tune':
        pass
    return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # main
    result = main(project_parameters=project_parameters)
