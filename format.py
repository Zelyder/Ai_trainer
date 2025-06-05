
def generate_recommendations(ideal_pose, real_pose):
    angle_points = [
        (11, 13, 15), # Левый локоть
        (12, 14, 16), # Правый локоть
        (23, 25, 27), # Левое колено
        (24, 26, 28)  # Правое колено
    ]

    recommendations = []
    for i, j, k in angle_points:
        ideal_angle = calculate_angle(ideal_pose[i], ideal_pose[j], ideal_pose[k])
        real_angle = calculate_angle(real_pose[i], real_pose[j], real_pose[k])
        if abs(ideal_angle - real_angle) > 10:  # Допустимое отклонение
            recommendations.append(f"Корректируйте угол между точками {i}-{j}-{k}. Эталонный:"
                                   f" {ideal_angle:.2f}, ваш: {real_angle:.2f}")
    return recommendations
# Пример использования
ideal_pose = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]
real_pose = [(0.55, 0.5), (0.65, 0.4), (0.75, 0.35)]

recommendations = generate_recommendations(ideal_pose, real_pose)
for rec in recommendations:
    print(rec)


